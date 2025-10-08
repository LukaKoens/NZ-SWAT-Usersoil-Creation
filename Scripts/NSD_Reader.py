from NSD_dataset_preprocessing import Read_NSD_data
import pandas as pd

hrzn_1 = "../data/sd_horizon_data.csv"
hrzn_2 = "../data/sd_horizon.csv"

obs_1 = "../data/ob_observation_data.csv"

soil = "../data/sd_soil.csv"

NSD_data = Read_NSD_data(hrzn_1, hrzn_2, obs_1, soil)

NSD_data.to_csv("../intermediate_data/NSD_processed_data.csv", index=False)


## The most important variable for soil mapping is total carbon percent.
## Filter out rows with missing total carbon percent values.
Useable_NSD_data = NSD_data.dropna(subset=["total_carbon_percent"])

## There are some negative values in horzion depth, which descrbie leaf liter and top vegetation layers,
#  this data is kinda weird to work with and causes problems.

Useable_NSD_data = Useable_NSD_data[Useable_NSD_data["horizondepth_maxval"].astype(float) > 0]

Useable_NSD_data.to_csv("../intermediate_data/Useable_NSD_data.csv", index=False)