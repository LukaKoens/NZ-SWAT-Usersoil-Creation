from NSD_dataset_preprocessing import ConvertNSD_To_ML_readable
import pandas as pd
import numpy as np
import ydf


def Predict_data(full_usable_data, sol_ksat_set):
    
    ## Define the inital and then secondary columns that I'll be trying to predict
    # for the secondary prediction targets, the results from the first prediction will be included
    target_cols = ['clay_percent', 'silt_percent', 'sand_percent', 'rock_fragment_percent']
    secondary_target_cols = ['whole_bulk_density', 'available_water_capacity']

    ## I found that the model decided that the ID was the most import variable for predicting.
    # As I've already explored the accuracy and reliablity of these models in my inital testing I didn't include a testing subset. 
    predictor_cols = [col for col in full_usable_data.columns if col not in target_cols and col not in secondary_target_cols and col != 'sd_horizon_id' and col != "site_identifier"]


    ## Creating the Actual models, I drop any Na's in the target column, this means each model will be trained on a slightly different dataset.
    cl_model = ydf.GradientBoostedTreesLearner(label="clay_percent", task=ydf.Task.REGRESSION).train(full_usable_data[predictor_cols + ['clay_percent']].dropna(subset='clay_percent'))
    si_model = ydf.GradientBoostedTreesLearner(label="silt_percent", task=ydf.Task.REGRESSION).train(full_usable_data[predictor_cols + ['silt_percent']].dropna(subset='silt_percent'))
    sa_model = ydf.GradientBoostedTreesLearner(label="sand_percent", task=ydf.Task.REGRESSION).train(full_usable_data[predictor_cols + ['sand_percent']].dropna(subset='sand_percent'))
    rk_model = ydf.GradientBoostedTreesLearner(label="rock_fragment_percent", task=ydf.Task.REGRESSION).train(full_usable_data[predictor_cols + ['rock_fragment_percent']].dropna(subset='rock_fragment_percent'))

    ## make a copy to apply the models too
    predicted_nsd = full_usable_data.copy()

    ## Predict each column and clip them to approriate values to prevent any outliers. ( might be worth round aswell as often there will be very low fractions)
    predicted_nsd['clay_percent'] = cl_model.predict(predicted_nsd)
    predicted_nsd['clay_percent'] = np.clip(predicted_nsd['clay_percent'], 0, 100)

    predicted_nsd['silt_percent'] = si_model.predict(predicted_nsd)
    predicted_nsd['silt_percent'] = np.clip(predicted_nsd['silt_percent'], 0, 100)

    predicted_nsd['sand_percent'] = sa_model.predict(predicted_nsd)
    predicted_nsd['sand_percent'] = np.clip(predicted_nsd['sand_percent'], 0, 100)

    predicted_nsd['rock_fragment_percent'] = rk_model.predict(predicted_nsd)
    predicted_nsd['rock_fragment_percent'] = np.clip(predicted_nsd['rock_fragment_percent'], 0, 100)


    ## Create a copy of the data and then fill in any NA values based on the predicited values.
    full_data_filled = full_usable_data.copy()
    full_data_filled = full_data_filled.fillna(predicted_nsd[['clay_percent', 'silt_percent', 'sand_percent', 'rock_fragment_percent']])

    ## Becuase I found that the Bulk density was more reliablly predicited than avaliable water capacity, I predict that first and then use that result to help infer the avaliable water capacity.
    
    bsd_model = ydf.GradientBoostedTreesLearner(label="whole_bulk_density", task=ydf.Task.REGRESSION).train(full_data_filled[predictor_cols + target_cols + ['whole_bulk_density']].dropna(subset='whole_bulk_density'))

    predicted_nsd['whole_bulk_density'] = bsd_model.predict(predicted_nsd)
    full_data_filled = full_data_filled.fillna(predicted_nsd[['whole_bulk_density']])

    
    ## Use all data both real and predicted to determine the avaliable water capacity. 
    
    awc_model = ydf.GradientBoostedTreesLearner(label="available_water_capacity", task=ydf.Task.REGRESSION).train(full_data_filled[predictor_cols + target_cols + secondary_target_cols].dropna(subset='available_water_capacity'))
    predicted_nsd['available_water_capacity'] = awc_model.predict(predicted_nsd)

    full_data_filled = full_data_filled.fillna(predicted_nsd[['available_water_capacity']])


    ## Predict the Saturated hyrdualic conductivity based on the global dataset.

    ksat_model = ydf.GradientBoostedTreesLearner(label="ksat_lab", task=ydf.Task.REGRESSION).train(sol_ksat_set)
    full_data_filled['ksat'] = ksat_model.predict(predicted_nsd)


    return full_data_filled

