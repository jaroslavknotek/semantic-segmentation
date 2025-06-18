import torch
import semseg.dataset as ds
import numpy as np
from segmentation_models_pytorch import Unet
from semseg.training import FocalLoss,DiceLoss
from torch.nn import BCELoss
import os
import logging
from tqdm.cli import tqdm
import os

import semseg.exp as exp
from tqdm.contrib.logging import logging_redirect_tqdm
from semseg.training import train
import semseg.prediction as prd

from types import SimpleNamespace

import semseg.exp as exp
from tqdm.contrib.logging import logging_redirect_tqdm
from semseg.training import train
import semseg.prediction as prd
import semseg.instance_matching as im
import semseg.imageutils as iu


def create_dataloaders(
    data_params,
    imgs,
    labels
):
    patch_size = data_params.patch_size
    x_channels = data_params.x_channels
    
    train_augumentation_fn = ds.setup_augumentation(
        patch_size = patch_size,
        blur_sharp_power=1,
        flip_horizontal=False,
        flip_vertical=False,
        #noise_value=0.01,
        rotate_deg=10,
        erasing = False
    )
    val_augumentation_fn = ds.setup_augumentation(patch_size = patch_size)

    train_dl,val_dl = ds.prepare_dataloaders(
        imgs,
        labels,
        train_augumentation_fn,
        val_augumentation_fn,
        batch_size= 32,
        val_size = 0.25,
        x_channels=x_channels
    )
    
    return (train_dl, val_dl)

def prepare_model(model_params):
    decoder_channels = np.array([model_params.starting_decoder_channel]*model_params.encoder_depth)
    pows = list(range(model_params.encoder_depth))
    decoder_channels = decoder_channels * [2**p for p in pows]
    
    return Unet(
        encoder_name=model_params.encoder_name,
        in_channels=model_params.x_channels,
        classes=3,
        activation="sigmoid",
        encoder_depth = model_params.encoder_depth,
        decoder_channels = decoder_channels, 
        decoder_use_batchnorm=True,
    )

def resolve_loss(loss_name):
    match loss_name:
        case "bce":
            return BCELoss()
        case "dice":
            return DiceLoss()
        case "fl":
            return FocalLoss(alpha = .9,gamma=.8)
        case _:
            raise ValueError(f"invalid loss {loss_name=}")
            


def run_training_loop(
    data_params,
    model_params,
    training_params,
    device,
    loaded_images,
    test_thrs = [.25,.5,.75],
    use_tqdm=True
):  
    model_path = training_params.model_path
    checkpoint_path = model_path.parent/f"checkpoint-{model_path.name}"
    
    (imgs, labels),(test_imgs,test_labels) = loaded_images
    train_dl, val_dl = create_dataloaders(data_params,imgs,labels,use_tqdm = use_tqdm)
    model = prepare_model(model_params)
    loss = resolve_loss(training_params.loss_name)
    
    with logging_redirect_tqdm():
        train_loss,val_loss = train(
            model,
            train_dl,
            val_dl,
            loss,
            checkpoint_path=checkpoint_path,
            patience=training_params.patience,
            scheduler_patience= training_params.scheduler_patience,
            epochs=training_params.epochs,
            device = device,
            lr = .001,
            use_tqdm = use_tqdm
        )

    test_preds = prd.segment_many(
        model,
        test_imgs,
        device = device,
        x_channels=data_params.x_channels,
        use_tqdm = use_tqdm
    )
    results = {}
    for thr in test_thrs:
        mean_precision,    mean_recall,    mean_f1 = im.measure_labeled(
            thr,
            test_labels,
            test_preds,
            data_params.small_filter,
            component_limit=1000,
            use_tqdm = use_tqdm,
        )
        results[f"{thr=:.2f}"] = {
            "mean_precision":mean_precision,
            "mean_recall":mean_recall,
            "mean_f1":mean_f1,
        }

    return results, model, test_preds, test_imgs,test_labels

def run_experiment(
    model_dir, 
    loss_name,
    starting_filters, 
    n_layers, 
    patch_size,
    small_filter, 
    loaded_images, 
    test_thrs = [.25,.5,.75],
    use_tqdm = False, 
    device = None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    x_channels = 1
    
    model_stem = param_to_str(loss=loss_name,sf=starting_filters,l=n_layers,ps=patch_size)
    model_path = model_dir/f"ds-{model_stem}.pth"
    logging.info(f"Training: {model_stem}")
    
    try:
        os.remove(model_path)
    except FileNotFoundError:
        pass # thats fine

    model_params = SimpleNamespace(**{
        "starting_decoder_channel": 16,
        "encoder_depth":1,
        "x_channels" : x_channels,
        "encoder_name":"resnet18",
    })
    data_params = SimpleNamespace(**{
        "patch_size" :patch_size,
        "x_channels" : x_channels,
        "small_filter": small_filter,
    })
    training_params = SimpleNamespace(**{
        "model_path": model_path,
        "loss_name":"bce",
        "epochs":None,
        "patience": 20,
        "scheduler_patience":10,
        # "epochs":1,
        # "patience": None,
        # "scheduler_patience":None,
    })
    
    results, model, test_preds, test_imgs,test_labels =  exp.run_training_loop(
        data_params,
        model_params,
        training_params,
        device,
        loaded_images,
        test_thrs = test_thrs,
        use_tqdm = use_tqdm
    )
    model_path.parent.mkdir(exist_ok=True,parents=True)
    torch.save(model, model_path)
    
    return results, model, test_preds, test_imgs,test_labels

def param_to_str(**kwargs):
    pairs = [f"{k}_{v}" for k,v in sorted(kwargs.items(),key = lambda x:x[0])]
    return '-'.join(pairs)