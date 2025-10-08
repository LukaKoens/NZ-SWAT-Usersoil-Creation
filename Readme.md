## Mapping the NSD to the FSL for the creation of a SWAT compatiable usersoil dataset

this colleciton of scripts takes in the NSD .csv dataset, and the SoilKsatDB database to produce / derive and complete values using a Yggdrasil Decision Forest regresion model to fill in blanks.

While the NSD data can be viewed freely via the NSD explorer at https://viewer-nsdr.landcareresearch.co.nz/search, this ommits some key information such as the bulk density and a number of other key values that are essential in the modeling process. As such I've included the tables needed to for this process from the NSD in the data. 

Many of the key points in this process are derived from the process outlined by Parshotam, 2018 ( https://environment.govt.nz/assets/OIA/Files/20-D-02513_0.pdf ).
This includes the mapping for missing NZSC codes found in the FSL but not the NSD, as well as the site selection used in mapping the NZSC codes to NSD samples, aswell as helping in formulate much of the underlying process and logic.
These tables are provided as the FSL_NZSC_NSD_SITE_MAP.csv and the MissingNZSC_Replacements.csv, and are based on tables 8 and 7 respectively from Parshotam, 2018, however there are some minor changes to the site selection in FSL_NZSC_NSD_SITE_MAP due to the provided sites being malformed in the NSD and not being present in the NSD explorer, so alternative selections were made based on the NZSC code for the soil. 

After collating and cleaning the bulk of the NSD data, Regression models are derived using the radnom forest models provided by the YDF python package ( https://ydf.readthedocs.io/en/stable/#next-steps ), which builds on from the deprecited tensor flow random forest model. 

Missing fine earth values are predicted as well as missing rock percentages, follwoing this, the bulk density and avaliable water capacity are predicted, using data from within the NSD.

Saturated hyrdaulic conductivity is missing from the NSD, as such the external dataset SoilKsatDB from Gupta, S. et al, 2021 ( https://doi.org/10.5194/essd-13-1593-2021 ) is used to fill in these gaps, by building a similar Regression model based on this data set, it can be applied to the data present in the NSD, initally I inteded to use the equations outlined by Saxton and Rawls, 2006, for the SPAW tool, however given their method I assumed basing it off a more complex dataset thats been made avaliable since would produce better results.

Due to time constraints and project requirements, there has been limited analysis and in depth evauluation on the results from these predictions, however especially for the fine earth percentages the derived models had quite low residual error, but for the more complex values such as saturated hydraulic conducitivity and avaliable water capacity, the models produced might not be adequate, however bulk density was on par with the fine earth models. 


## References

- Parshotam, 2018. New Zealand SWAT: Deriving a New Zealand-wide soils dataset from the NZ-NSD for use in the Soil and Water Assessment Tool (SWAT). Report prepared for Ministry for the Enviroment. Project: WL18033. Aqualinc Research Limited. https://environment.govt.nz/assets/OIA/Files/20-D-02513_0.pdf

- Manaaki Whenua - Landcare Research 2020. National Soils Database (NSD). https://doi.org/10.26060/95m4-cz25

- Gupta, S., Hengl, T., Lehmann, P., Bonetti, S., and Or, D.: SoilKsatDB: global database of soil saturated hydraulic conductivity measurements for geoscience applications, Earth Syst. Sci. Data, 13, 1593–1612, https://doi.org/10.5194/essd-13-1593-2021, 2021. 

- https://ydf.readthedocs.io/en/stable/#next-steps

- Saxton, K. E., and W. J. Rawls. “Soil Water Characteristic Estimates by Texture and Organic Matter for Hydrologic Solutions.” Soil Science Society of America Journal, vol. 70, no. 5, 2006, p. 1569, https://doi.org/10.2136/sssaj2005.0117.


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


## Running the Scripts

This process requires 3* python packages,
    - pandas
    - geopandas
    - ydf 
    - *( ipykernel for running the notebook )

A number of the larger functions have been saved under the scripts folder, this is to improve the readability of the main process in the notebook,
running the notebook show produce a usersoil dataset, however the lookup table will need to be made afterwards, keep in mind the mapping for the different NZSC codes, however using geopandas you can add these mapped values to the shape file.

if you have any question feel free to get in touch at luka.koens@gmail.com.




