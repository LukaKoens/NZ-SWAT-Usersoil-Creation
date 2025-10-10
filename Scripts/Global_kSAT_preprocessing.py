import pandas as pd


## The soil saturated hydraulic conducitivity dataset from https://essd.copernicus.org/articles/13/1593/2021/ 

def preprocess_ksat_data(sol_ksat_df, outlier_threshold=2000):
    
    ## Drop all the columns that have na values as they are unsuitable for training off.
    sol_ksat_df = sol_ksat_df.dropna(subset=["clay_tot_psa", "sand_tot_psa", "silt_tot_psa", "db","hzn_top","hzn_bot","oc_v", "w3cld","w15l2", "ksat_lab"])

    ## pick out the columns of interest
    sol_ksat_training_set = sol_ksat_df[["hzn_top","hzn_bot", "clay_tot_psa", "sand_tot_psa", "silt_tot_psa","oc_v", "db","w3cld","w15l2", "ksat_lab"]]

    ## convert the field capacity and permant wilt point to avaliable water capacity then drop them
    sol_ksat_training_set["awc"] = (sol_ksat_training_set["w3cld"] - sol_ksat_training_set["w15l2"])


    print(sol_ksat_training_set["ksat_lab"].mean())
    sol_ksat_training_set["ksat_lab"] = ( sol_ksat_training_set["ksat_lab"] * 10 ) / 24.0 

    sol_ksat_training_set = sol_ksat_training_set.drop(columns=["w3cld","w15l2"])

    print(sol_ksat_training_set["ksat_lab"].mean())

    ## Remove extereme outliers
    sol_ksat_training_set = sol_ksat_training_set[sol_ksat_training_set["ksat_lab"] < outlier_threshold]


    ## Rename the columns to match the NSD dataset
    sol_ksat_training_set.rename(columns={
        "hzn_top": "horizondepth_maxval",
        "hzn_bot": "horizondepth_minval",
        "clay_tot_psa": "clay_percent",
        "sand_tot_psa": "sand_percent",
        "silt_tot_psa": "silt_percent",
        "oc_v": "total_carbon_percent",
        "db": "whole_bulk_density",
        "awc": "available_water_capacity",
        }, inplace=True)


    return sol_ksat_training_set