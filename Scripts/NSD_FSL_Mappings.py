import pandas as pd
from calc_usle_k import calc_usle_k




## Pick out the Sites in the NSD that are actually referenced 

    ## may not actuall need the sa_site table 


## Get the NSD horizion layers for the sites that are actually referenced



## Create the Headers for the usersoil dataset. 

## Find the number of sites, and the maximum numnber of layers of any of them.
SITE_NLAYERS = nsd_data["site_identifier"].value_counts()
MAX_LAYERS = sig_nsd_layers["horizonnumber"].max().astype(int)

## Two sets of headers, one which is per site, and one which is per layer per site.

primary_headers = ["OBJECTID","SITE_ID_SB","SITE_ID","MUID","SEQN","SNAM","S5ID","CMPPCT","NLAYERS","HYDGRP","SOL_ZMX","ANION_EXCL","SOL_CRK","TEXTURE"]
layer_headers = ["LAYER_ID#", "SOL_Z#", "SOL_BD#", "SOL_AWC#", "SOL_K#", "SOL_CBN#", "CLAY#", "SILT#", "SAND#", "ROCK#", "SOL_ALB#","USLE_K#","SOL_EC#"]

## Given the maximum number of layers, populate the layer headers acordingly.

layer_headers_renamed = [header.replace("#", str(i)) for i in range(1, MAX_LAYERS+1) for header in layer_headers]

## make the data frame

usersoil_headers = primary_headers + layer_headers_renamed
usersoil = pd.DataFrame(columns=usersoil_headers, index=range(len(SITE_NLAYERS)))


## Map the Per Site Columns to target NSD sites ( uses the NSD_data data not the sa_site data )

usersoil["SITE_ID"] = sig_nsd_layers["sa_site_id"].unique().tolist()  


sig_ob_ids = sig_nsd_layers[["site_identifier", "sa_site_id"]].drop_duplicates().set_index("sa_site_id").to_dict()["site_identifier"]
sig_ob_nlayers = sig_nsd_layers[["sa_site_id", "horizonnumber"]].groupby("sa_site_id").max().to_dict()["horizonnumber"]
sig_ob_class = sig_nsd_layers[["sa_site_id", "classifier_nzsc"]].set_index("sa_site_id").to_dict()["classifier_nzsc"]
sig_ob_max_depth = sig_nsd_layers[["sa_site_id", "horizondepth_maxval"]].groupby("sa_site_id").max().to_dict()["horizondepth_maxval"]


usersoil["SITE_ID_SB"] = usersoil["SITE_ID"].map(sig_ob_ids)
usersoil["NLAYERS"] = usersoil["SITE_ID"].map(sig_ob_nlayers)

## Sets the SNAM as the NZSC code and strips off the long hand name
usersoil["SNAM"] = usersoil["SITE_ID"].map(sig_ob_class)
usersoil["SNAM"] = usersoil["SNAM"].str.split().str[0]

## Gets the max depth and converts it to mm
usersoil["SOL_ZMX"] = usersoil["SITE_ID"].map(sig_ob_max_depth)
usersoil["SOL_ZMX"] = usersoil["SOL_ZMX"] * 10

usersoil["OBJECTID"] = range(1, len(usersoil)+1)


## Create a dict of all the horizion ids, based on the site id and horizion number
sig_ob_layer_ids = sig_nsd_layers[["sd_horizon_id", "horizonnumber", "sa_site_id"]].set_index(["sa_site_id", "horizonnumber"]).to_dict()["sd_horizon_id"]

## Becuase of the preprocessing steps, this doesn't need to handle any errors in the data, such as soil layers above depth 0, however this should be kept in mind if issues arise.

##  Its not best practice to iterate over pandas rows, but this isn't a high frequency operation.

## Becuase, there are often layers, either above ground ( leaf litter etc ), that I've ommited, the horizonnumber value, doesn't always line up with what is present
## This accounts for that and adjusts the data accordingly before mapping the actual values in.
for row in usersoil.itertuples():
    site_id = row.SITE_ID
    missing_layers = 0
    i = 1
    while i < row.NLAYERS + 1:
        if (site_id, i) in sig_ob_layer_ids:
            usersoil.at[row.Index, f"LAYER_ID{i-missing_layers}"] = sig_ob_layer_ids.get((site_id, i), None)
            print(f"Mapping layer {i} for site {site_id} to horizon id {usersoil.at[row.Index, f'LAYER_ID{i}']}, for Layer {i-missing_layers}")
            i += 1
        else:
            missing_layers += 1
            i += 1
            print(f"Skipping missing or shallow layer {i} for site {site_id}")
    
    usersoil.at[row.Index, "NLAYERS"] = row.NLAYERS - missing_layers
    

## Maps the data both predicted and measured from the NSD into the usersoil data    
for row in usersoil.itertuples():
    
    site_id = row.SITE_ID
    
    for i in range(1, row.NLAYERS + 1):
        layer_id = getattr(row, f"LAYER_ID{i}")
        if pd.notna(layer_id):
            
            ## in SWAT, fine earth, and rocks all need to add up to 100%, in the NSD fine earth is given as a percantage indepent of rock, so this needs to be adjusted for here.
            ## this also helps to smooth any odd values produced during the predicting process bring thing back to a normal range ( tho maybe over represented ? )
            usersoil.at[row.Index, f"ROCK{i}"] = nsd_data['rock_fragment_percent'].get(layer_id, None)
            usersoil.at[row.Index, f"CLAY{i}"] = ((1.0 - usersoil.at[row.Index, f"ROCK{i}"] / 100.0 ) / 1.0 ) * nsd_data["clay_percent"].get(layer_id, None)
            usersoil.at[row.Index, f"SILT{i}"] = ((1.0 - usersoil.at[row.Index, f"ROCK{i}"] / 100.0 ) / 1.0 ) * nsd_data["silt_percent"].get(layer_id, None)
            usersoil.at[row.Index, f"SAND{i}"] = ((1.0 - usersoil.at[row.Index, f"ROCK{i}"] / 100.0 ) / 1.0 ) * nsd_data["sand_percent"].get(layer_id, None)

            usersoil.at[row.Index, f"SOL_BD{i}"] = nsd_data["whole_bulk_density"].get(layer_id, None)
            usersoil.at[row.Index, f"SOL_AWC{i}"] = nsd_data["available_water_capacity"].get(layer_id, None)
            usersoil.at[row.Index, f"SOL_CBN{i}"] = nsd_data["total_carbon_percent"].get(layer_id, None)
            usersoil.at[row.Index, f"SOL_K{i}"] = nsd_data["ksat"].get(layer_id, None)

            usersoil.at[row.Index, f"SOL_ALB{i}"] = nsd_data['CLR_alb'].get(layer_id, None)
            usersoil.at[row.Index, f"USLE_K{i}"] = calc_usle_k(usersoil.at[row.Index, f"SOL_CBN{i}"], usersoil.at[row.Index, f"SAND{i}"], usersoil.at[row.Index, f"SILT{i}"], usersoil.at[row.Index, f"CLAY{i}"])


usersoil.to_csv("../nzsc_usersoil.csv")