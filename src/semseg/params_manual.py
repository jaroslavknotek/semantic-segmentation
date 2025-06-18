import json
import itertools
from tqdm.cli import tqdm
import semseg.exp as exp
from pathlib import Path
from tqdm.contrib.logging import logging_redirect_tqdm
import traceback

import logging

logger = logging.getLogger('param_search_manual')

def grid_search(params,loaded_images,output_root,model_dir,small_filter = 0,n_runs = 1, test_thrs = [.25,.5,.75], use_tqdm = False):
    output_root = Path(output_root)
    
    with logging_redirect_tqdm():
        for exp_params in tqdm(
            params,
            desc = 'Running experiments',
            disable = not use_tqdm
        ):
            n_layers = exp_params['layers']
            start_filter = exp_params['start_filter']
            patch_size = exp_params['patch_size']
            loss_name = exp_params['loss']

            
            
            
            for i in range(n_runs):
                if n_runs>1:
                    model_stem = exp.param_to_str(l=n_layers,sf=start_filter,ps=patch_size,loss=loss_name,run=i+1)
                else:
                    model_stem = exp.param_to_str(l=n_layers,sf=start_filter,ps=patch_size,loss=loss_name)
                    
                logger.info(f"Processing {model_stem}")
                output_path = output_root/f"{model_stem}.txt"
                if output_path.exists():
                    continue
                try:
                    results, model, test_preds, test_imgs,test_labels = exp.run_experiment(
                        model_dir,
                        loss_name, 
                        start_filter, 
                        n_layers, 
                        patch_size,
                        small_filter,
                        loaded_images,
                        test_thrs = test_thrs,
                        use_tqdm= False
                    )
                except Exception as e:
                    exc = traceback.format_exc()
                    results = {"error":exc}
                    raise e


                output_path.parent.mkdir(exist_ok=True, parents=True)
                with open(output_path,'w') as f:
                    json.dump(results,f)


def get_products(params_arrays):
    keys = params_arrays.keys()
    vals = list(params_arrays.values())
    return list([{k:v for k,v in zip(keys,prods)} for prods in itertools.product(*vals)])