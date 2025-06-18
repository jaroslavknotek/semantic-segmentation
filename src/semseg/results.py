import matplotlib.pyplot as plt
import semseg.io as io
import semseg.visualisation as vis
import numpy.typing as npt
from dataclasses import dataclass
import pandas as pd


UM = 10**(-6)


@dataclass
class SegmentationResults:
    img:npt.NDArray
    individual_precipitates:list
    features:list
    prediction:npt.NDArray
    pixel_size:float or None

def save_results(results_dir,results):
    name = results_dir.name
    
    image_width = results.img.shape[1]
    io.image_write(results_dir/'img.png',results.img)
    io.image_write(results_dir/'prediction_continous.png',results.prediction)
    
    # plot
    fig = vis.plot_particles(
        results.img,
        results.individual_precipitates,
        name,
    )
    fig.savefig(results_dir/'plot_visualized.png')
    plt.close(fig)
    
    # features
    ## hist areas px
    areas = [f.area_px for f in results.features]
    fig,ax = plt.subplots(1,1)
    ax.hist(areas,bins = 100)
    ax.set_title(name)
    fig.savefig(results_dir/'histogram_px.png')
    plt.close(fig)
    
#     ## hist areas um
#     if results.pixel_size:
#         areas_um = [f.area_px / results.pixel_size for f in results.features]
#         fig,ax = plt.subplots(1,1)
#         ax.hist(areas_um,bins = 100)
#         ax.set_title(name)
#         fig.savefig(results_dir/'histogram_um.png')
#         plt.close(fig)
    
    
    df_bb = pd.DataFrame([ind.bounding_box for ind in results.individual_precipitates])
    df_features = pd.DataFrame(results.features)
    df = pd.concat([df_bb,df_features],axis = 1)
    if results.pixel_size:
        df['pixel_size'] = results.pixel_size
        # converting square units works bit different, hence the following
        square_pixel_area_um = convert_px2um(1,df['pixel_size'])**2
        df['area_um'] = square_pixel_area_um*df['area_px']
        
        cols = ['feret_min','feret_max','feret_90']
        for c in cols:
            df[f"{c}_um"] = convert_px2um(df[f"{c}_px"],df['pixel_size'])

    csv_path = results_dir/'precipitates.csv'
    df.to_csv(csv_path,header=True,index=False)
    
    
def convert_px2um(px,pixel_size):
    return pixel_size / (UM/px)