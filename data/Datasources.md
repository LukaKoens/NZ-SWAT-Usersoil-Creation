## Data Sources

This process uses data from 4 sources. due licesning & copyright I've ommited the underlying data and provide just the scripts used to maniplute it, bellow is a descripitoin of how I've formated and stored the data to get it to work within the proces provided.

The main data is the NSD data, this needs to be requested from Manaaki Whenua, there is a download page here: https://viewer-nsdr.landcareresearch.co.nz/datasets/downloads/1042-2, however this won't work, and I found I needed to contact them directly via https://viewer-nsdr.landcareresearch.co.nz/contact-us and after sometime, I recived a download link for both the CSV and Database form of the NSD, in this case the CSV form is the one used.

In this project the following csv files are needed from the NSD.

project
    - data
        -   ob_observation_data.csv
        -   sd_horizon_data.csv
        -   sd_horizon.csv
        -   sd_soil.csv

The FSL layer is used as the spatial component of the dataset, in some locations the S-Map is avaliable and likely a better choice, but not for my case so I didn't explore it much.
I just store it in it's own file, it's referenced half way down the NSD_soil_mapping notebook.

project 
    - lris-fsl-north-island-v11-all-attributes-SHP
        -  fsl-north-island-v11-all-attributes.shp

For the FSL mapping, two tables are needed, these are both described by Parshotam, 2018, in the paper New Zealand SWAT: Deriving a New Zealand-wide soils dataset from the NZ-NSD for use in the Soil and Water Assessment Tool (SWAT). https://environment.govt.nz/assets/OIA/Files/20-D-02513_0.pdf in tables 7 and 8.

I manually converted these to text files and then into csv's using pandas, these are used to fix some missing NZSC values ( table 7 ) and then the site selection for each NZSC code ( table 8 )

project
    - data
        -   FSL_NZSC_NSD_SITE_MAP.csv
        -   MissingNZSC_Replacements.csv


finally I used the sol_ksat.pnts_horizons.csv file from the SoilKsatDB, https://doi.org/10.5194/essd-13-1593-2021

project
    - data
        -   sol_ksat.pnts_horizons.csv


all intermediate data products are stored under " intermediate_data "
the final output is saved into the parent directory.

