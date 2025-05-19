import torch
import itertools
import numpy as np
from torch.functional import F
from pathlib import Path
from tqdm.auto import tqdm
import logging
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger("semseg.training")


class StopTrainingException(Exception):
    pass


class EmptyCallable:
    def __call__(self):
        return None


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss):
        return self.early_stop(validation_loss)

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            logger.debug(
                f"Early stopper's patience {self.patience=} update. ({self.counter})"
            )
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopper's {self.patience=} run out with {self.min_validation_loss=:.5}"
                )
                raise StopTrainingException()


class MetricCheckpointer:
    def __init__(self, model, model_path, min_delta=0):
        self.model = model
        model_path = Path(model_path)
        model_path.parent.mkdir(exist_ok=True, parents=True)
        self.model_path = model_path
        self.min_validation_loss = np.inf
        self.min_delta = min_delta

    def __call__(self, validation_loss):
        return self.checkpoint_if_best(validation_loss)

    def checkpoint_if_best(self, validation_loss):
        if self.model_path is None:
            return False

        if validation_loss > self.min_validation_loss + self.min_delta:
            return False

        before_val_loss = self.min_validation_loss
        self.min_validation_loss = validation_loss

        best_val_loss = self.min_validation_loss
        logger.info(
            f"Improvement {before_val_loss=:.5} -> {best_val_loss:.5}! Checkpointing model!"
        )

        torch.save(self.model, self.model_path)
        return True


class DiceLoss:
    def __init__(self, smooth=1):
        self.smooth = smooth
    
    def __call__(self, pred, target):
        #pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    
class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=0.8,
        gamma=2,
        reduction="mean",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        bce_exp = torch.exp(-bce)
        return self.alpha * (1 - bce_exp) ** self.gamma * bce


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    epochs=None,
    patience=None,
    scheduler_patience=None,
    checkpoint_path=None,
    lr=0.001,
    device="cpu",
    use_tqdm = False,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def step_fn(targets):
        return step(model, targets, loss_fn, device=device)

    def train_epoch_fn():
        return train_epoch(model, train_dataloader, optimizer, step_fn)

    def eval_epoch_fn():
        return validate_epoch(model, val_dataloader, step_fn)

    after_callbacks = []
    if scheduler_patience is not None:
        after_callbacks += [setup_scheduler(optimizer, scheduler_patience)]

    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    if patience is not None:
        after_callbacks += [early_stopper]

    if checkpoint_path is not None:
        after_callbacks += [MetricCheckpointer(model, checkpoint_path)]

    train_losses = []
    validation_losses = []
    epochs_iter = range(epochs) if epochs is not None else itertools.count()
    with logging_redirect_tqdm(),tqdm(epochs_iter, desc="Training epochs",disable= not use_tqdm) as t:
        for epoch in t:
            try:
                loss_train, loss_val = run_epoch(
                    train_epoch_fn, eval_epoch_fn, after_callbacks
                )
                train_losses.append(loss_train)
                validation_losses.append(loss_val)

                t.set_postfix(
                    epoch=epoch,
                    validation_loss=loss_val,
                    patience = f"{early_stopper.counter}/{patience}"
                )
            except StopTrainingException:
                break

    return train_losses, validation_losses


def train_epoch(model, dataloader, optimizer, step_fn):
    model.train()

    def train_step(targets):
        optimizer.zero_grad()
        ls = step_fn(targets)
        ls.backward()
        optimizer.step()
        return ls.item()

    losses = [train_step(targets) for targets in dataloader]
    return np.mean(losses)


def validate_epoch(model, dataloader, step_fn):
    model.eval()
    with torch.no_grad():
        losses = [step_fn(t).item() for t in dataloader]
        return np.mean(losses)


def run_epoch(
    train_epoch_fn,
    validate_epoch_fn,
    after_callbacks,
):
    train_loss = train_epoch_fn()
    val_loss = validate_epoch_fn()

    for callback in after_callbacks:
        callback(val_loss)
    return train_loss, val_loss


def step(model, targets, loss_fn, device="cpu"):
    device_targets = {k: v.to(device) for k, v in targets.items()}
    x = device_targets["x"]
    pred = model(x)

    y = device_targets["y"]
    return loss_fn(pred, y)


def setup_scheduler(optimizer, scheduler_patience):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=scheduler_patience
    )

    def scheduler_fn(validation_loss):
        scheduler.step(validation_loss)
        lr = _get_lr(optimizer)
        logger.debug(f"learning rate: {lr}")

    return scheduler_fn


def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
