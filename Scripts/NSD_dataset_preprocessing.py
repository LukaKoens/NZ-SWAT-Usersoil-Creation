import pandas as pd
import numpy as np
import re



## Preprocessing class for NSD dataset,
    ## This class handles the initial preprocessing of the NSD dataset,
        ## mapping and cleaning key columns from the raw data tables into a single dataframe.

class NSD_preprocessor():
    
    def __init__(self, horizion_1, horizion_2, obs, soil_data):
            self.hrzn_1 = pd.DataFrame(horizion_1)  ## This is the SD_HORIZON_DATA table
            self.hrzn_2 = pd.DataFrame(horizion_2)  ## This is the SD_HORIZON table ( there is a slight difference in columns present between these two tables)
            self.obs = pd.DataFrame(obs)            ## This is the OB_OBSERVATION_DATA table which contains the actual measurements for each horizon
            self.soil = pd.DataFrame(soil_data)     ## This is the SD_SOIL table which contains general site information mostly just used for the genetic classification of the soil profile.
            self.output = pd.DataFrame()

            self.key_columns_hrzn_1 = ["sd_horizon_id",
                                        "sd_soil_id",
                                        "horizonnumber" ,
                                        "designation",
                                        "horizondepth_maxval",
                                        "horizondepth_minval",
                                        "texture_qualifier",
                                        "texture_value",
                                        "texture_legacy_texturemodifier",
                                        "consistence_strength"]


    ## Map all the columns which can be directly mapped from the horizion tables to the output dataframe without any special processing.
    def direct_map(self) -> pd.DataFrame:
        
        self.output = self.hrzn_2[self.key_columns_hrzn_1].copy()
        self.output["horizonnumber"] = self.output["horizonnumber"].astype(float).astype(int)
        ## This whole process could likely be done in a more efficient way, but this is clear and works for now.

        temp_mapping = self.hrzn_1[["site_identifier", "sd_horizon_id"]].set_index("sd_horizon_id").to_dict()["site_identifier"]
        self.output["site_identifier"] = self.output["sd_horizon_id"].map(temp_mapping)

        temp_mapping = self.hrzn_1[["classifier_nzsc", "sd_horizon_id"]].set_index("sd_horizon_id").to_dict()["classifier_nzsc"]
        self.output["classifier_nzsc"] = self.output["sd_horizon_id"].map(temp_mapping)

        temp_mapping = self.hrzn_1[["profiledrainage", "sd_horizon_id"]].set_index("sd_horizon_id").to_dict()["profiledrainage"]
        self.output["profiledrainage"] = self.output["sd_horizon_id"].map(temp_mapping)

        temp_mapping = self.hrzn_1[["matrix_hue", "sd_horizon_id"]].set_index("sd_horizon_id").to_dict()["matrix_hue"]
        self.output["matrix_hue"] = self.output["sd_horizon_id"].map(temp_mapping)
        

        ## this is the most convienet time to calcuate the soil's albedo ratio so this is done here.
        # this needs to clean up the matrix value and chroma feilds as they are a bit of a mess.
        
        temp_mapping = self.hrzn_1[["matrix_value", "sd_horizon_id"]].set_index("sd_horizon_id").to_dict()["matrix_value"]
        self.output["matrix_value"] = self.output["sd_horizon_id"].map(temp_mapping)

        self.output["matrix_value"] = self.output["matrix_value"].str.replace(r"[;,/]", ",", regex=True)
        self.output["matrix_value"] = self.output["matrix_value"].str.split(",").str[0]

        self.output["matrix_value"] = pd.to_numeric(self.output["matrix_value"], errors="coerce")
        self.output["matrix_value"] = self.output["matrix_value"].fillna(0)
        
        temp_mapping = self.hrzn_1[["matrix_chroma", "sd_horizon_id"]].set_index("sd_horizon_id").to_dict()["matrix_chroma"]
        self.output["matrix_chroma"] = self.output["sd_horizon_id"].map(temp_mapping)

        self.output["matrix_chroma"] = self.output["matrix_chroma"].str.replace(r"[;,/]", ",", regex=True)
        self.output["matrix_chroma"] = self.output["matrix_chroma"].str.split(",").str[0]

        self.output["matrix_chroma"] = pd.to_numeric(self.output["matrix_chroma"], errors="coerce")
        self.output["matrix_chroma"] = self.output["matrix_chroma"].fillna(0)

        self.output["CLR_alb"] = self.output["matrix_value"].astype(int) / self.output["matrix_chroma"].astype(int)
        
        # Convert to float, set invalid to NaN
        self.output["CLR_alb"] = pd.to_numeric(self.output["CLR_alb"], errors="coerce")
        self.output["CLR_alb"] = self.output["CLR_alb"].replace(np.nan, 0)
        self.output["CLR_alb"] = self.output["CLR_alb"].replace(np.inf, 0)


        temp_mapping = self.soil[["classifier_nz_genetic", "sd_soil_id"]].set_index("sd_soil_id").to_dict()["classifier_nz_genetic"]
        self.output["classifer_nz_genetic"] = self.output["sd_soil_id"].map(temp_mapping)

        # Convert to string, replace separators with ','
        # Extract first numeric value
        
        
        # Extract first numeric value


        return self.output

    ## This function cleans up and standardizes the texture related columns in the dataset.
    def clean_texture_columns(self) -> pd.DataFrame:
        
        # this is a bit messy, but the data is messy too.

        for row in self.output.iterrows():
            idx, data = row
            # Fix texture_qualifier if missing
            # if pd.isna(self.output.at[idx, "texture_qualifier"]):
            val = self.output.at[idx, "texture_value"]
            
            if not isinstance(val, str):
                continue 
            if "very fine" in val:
                self.output.at[idx, "texture_qualifier"] = 4
                self.output.at[idx, "texture_value"] = val.replace("very fine ", "")              
            elif "fine" in val:
                self.output.at[idx, "texture_qualifier"] = 3
                self.output.at[idx, "texture_value"] = val.replace("fine ", "")
            elif "medium" in val:
                self.output.at[idx, "texture_qualifier"] = 2
                self.output.at[idx, "texture_value"] = val.replace("medium ", "")
            elif "very coarse" in val:
                self.output.at[idx, "texture_qualifier"] = 0
                self.output.at[idx, "texture_value"] = val.replace("very coarse ", "")
            elif "coarse" in val:
                self.output.at[idx, "texture_qualifier"] = 1
                self.output.at[idx, "texture_value"] = val.replace("coarse ", "")
            self.output.at[idx, "texture_value"] = self.output.at[idx, "texture_value"].strip()

            # Standardize texture_value codes
            val = self.output.at[idx, "texture_value"]
            if val == "ZL":
                self.output.at[idx, "texture_value"] = "silt loam"
            elif val == "SL":
                self.output.at[idx, "texture_value"] = "sandy loam"
            elif val == "LS":
                self.output.at[idx, "texture_value"] = "loamy sand"
            elif val == "S":
                self.output.at[idx, "texture_value"] = "sand"
            elif val == "CL":
                self.output.at[idx, "texture_value"] = "clay loam"
            elif val == "C":
                self.output.at[idx, "texture_value"] = "clay"
            elif val == "ZC":
                self.output.at[idx, "texture_value"] = "silty clay"
            elif val == "SC":
                self.output.at[idx, "texture_value"] = "sandy clay"
            elif val == "Z":
                self.output.at[idx, "texture_value"] = "silt"

            if "heavy" in val:
                self.output.at[idx, "texture_legacy_texturemodifier"] = "heavy"
                self.output.at[idx, "texture_value"] = val.replace("heavy ", "")

            elif "gravelly" in val:
                self.output.at[idx, "texture_legacy_texturemodifier"] = "gravelly"
                self.output.at[idx, "texture_value"] = val.replace("gravelly ", "")

            elif "peaty" in val:
                self.output.at[idx, "texture_legacy_texturemodifier"] = "peaty"
                self.output.at[idx, "texture_value"] = val.replace("peaty ", "")
            
            if "loem" in val:
                self.output.at[idx, "texture_value"] = val.replace("loem", "loam")     

        self.output["texture_qualifier"] = self.output["texture_qualifier"].replace({4: "very fine", 3: "fine", 2: "medium", 1: "coarse", 0: "very coarse"})
        self.output["consistence_strength"] = self.output["consistence_strength"].replace({"1": "very weak", "F": "firm", "3": "slightly firm", "2": "weak", "5": "very firm", "6":"hard","mod firm":"moderately firm", "mod weak":"moderately weak"})

        return self.output

    ## Map observation values to output dataframe based on sd_horizon_id
    def observation_mapping(self) -> pd.DataFrame:

        ## Each row in the observation table corresponds to a specific property measurement for a horizon.

        designation = self.obs[["sd_horizon_id","designation"]].drop_duplicates().set_index("sd_horizon_id").to_dict()["designation"]
        self.output["designation"] = self.output["designation"].fillna(self.output["sd_horizon_id"].map(designation))

        clay_values = self.obs[self.obs["propertytypename"] == "fe2" ].set_index("sd_horizon_id").to_dict()["result_val"]    ## The percent of soil particles which are < 0.002 mm (2 m) in equivalent diameter
        self.output["clay_percent"] = self.output["sd_horizon_id"].map(clay_values)

        silt_values = self.obs[self.obs["propertytypename"] == "silt" ].set_index("sd_horizon_id").to_dict()["result_val"]   
        self.output["silt_percent"] = self.output["sd_horizon_id"].map(silt_values)

        sand_values = self.obs[self.obs["propertytypename"] == "sand" ].set_index("sd_horizon_id").to_dict()["result_val"]
        self.output["sand_percent"] = self.output["sd_horizon_id"].map(sand_values)

        rock_values = self.obs[self.obs["propertytypename"] == "ws2mm" ].set_index("sd_horizon_id").to_dict()["result_val"]
        self.output["rock_fragment_percent"] = self.output["sd_horizon_id"].map(rock_values)   ## The percent of soil particles which are > 2 mm in equivalent diameter
        self.output["rock_fragment_percent"] = self.output["rock_fragment_percent"].apply(lambda x: 100 - float(x) if pd.notna(x) else x)  ## the rock fragment percent is the percent of particles < 2mm, so we need to convert it to from > 2mm

        carbon_values = self.obs[self.obs["propertytypename"] == "totalc" ].set_index("sd_horizon_id").to_dict()["result_val"]
        self.output["total_carbon_percent"] = self.output["sd_horizon_id"].map(carbon_values)   ## percent

        whole_bulk_density = self.obs[self.obs["propertytypename"] == "drybulkdensitywholesoil" ].set_index("sd_horizon_id").to_dict()["result_val"]  ## g/cm3
        self.output["whole_bulk_density"] = self.output["sd_horizon_id"].map(whole_bulk_density)

        mositure_content = self.obs[self.obs["propertytypename"] == "moisturefactor" ].set_index("sd_horizon_id").to_dict()["result_val"]  ## percent
        self.output["moisturefactor"] = self.output["sd_horizon_id"].map(mositure_content)

        avaliable_water_capacity = self.obs[self.obs["propertytypename"] == "totalavailablewater" ].set_index("sd_horizon_id").to_dict()["result_val"]  ## cm/cm
        self.output["available_water_capacity"] = self.output["sd_horizon_id"].map(avaliable_water_capacity)

        cec = self.obs[self.obs["propertytypename"] == "cec"].set_index("sd_horizon_id").to_dict()["result_val"]    ## used in regression equations to estimate Saturated Hydraulic Conductivity
        self.output["cec"] = self.output["sd_horizon_id"].map(cec)




    def get_data(self) -> pd.DataFrame:
        return self.output

def Read_NSD_data(horizion_1_path, horizion_2_path, obs_path, soil_data_path):
    horizion_1 = pd.read_csv(horizion_1_path, dtype=str)
    horizion_2 = pd.read_csv(horizion_2_path, dtype=str)
    obs = pd.read_csv(obs_path, dtype=str)
    soil_data = pd.read_csv(soil_data_path, dtype=str)

    # Convert numeric columns to appropriate types
    NSD_combined = NSD_preprocessor(horizion_1, horizion_2, obs, soil_data)
    NSD_combined.direct_map()
    NSD_combined.clean_texture_columns()
    NSD_combined.observation_mapping()

    return NSD_combined.get_data()

## Preprocessing class for ML, 
    ## This class contains static methods for preprocessing various columns in the NSD dataset,
        ## converting them into formats suitable for machine learning models.

class ML_preprocessor():
    def __init__(self):
        pass

    ## Parse soil designation into component flags,
        ## This is a bit ad-hoc, but should work for most cases, better refine later if data shows otherwise.
        ## during the training the regression model will complie that a number of these columns don't change, the specific columns change depedning the task, so there is some variance in them but it's very minor
    def parse_designation(desig):
        if not isinstance(desig, str) or desig.strip() == "":
            return {"DESG_prefix_num": 0, "DESG_master": "X", "DESG_subdivision": 0, "DESG_suffixes": [], "DESG_u_flag": 0}
        
        s = desig.strip()
        s = s.replace("(", "").replace(")", "")  # remove parentheses
        
        # Check for 'u' prefix
        u_flag = 0
        if s.startswith("u"):
            u_flag = 1
            s = s[1:]  # strip u
        
        # Find first number(s) at start as prefix
        prefix_match = re.match(r"^\d+", s)
        if prefix_match:
            prefix_num = int(prefix_match.group(0))
            s = s[prefix_match.end():]
        else:
            prefix_num = 0
        
        # Find first contiguous uppercase letters as master
        master_match = re.search(r"[A-Z]+", s)
        if master_match:
            master = master_match.group(0)
            rest = s[master_match.end():]
        else:
            master = "X"
            rest = s
        
        # Find trailing number as subdivision
        sub_match = re.search(r"\d+$", rest)
        if sub_match:
            subdivision = int(sub_match.group(0))
            suffix_part = rest[:sub_match.start()]
        else:
            subdivision = 0
            suffix_part = rest
        
        suffixes = list(suffix_part) if suffix_part else []
        
        return {
            "DESG_prefix_num": prefix_num,
            "DESG_master": master,
            "DESG_subdivision": subdivision,
            "DESG_suffixes": suffixes,
            "DESG_u_flag": u_flag
        }

    ## Preprocess texture_qualifier column, 
        ## This function cleans and encodes the `texture_qualifier` column into binary features, boiling down some of the detail to make it simpler.
    def preprocess_texture_qualifier(df: pd.DataFrame) -> pd.DataFrame:
        # 1. Normalize strings (lowercase, strip spaces)
        df["texture_qualifier"] = (
            df["texture_qualifier"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        # 2. Replace common typos / variants
        replacements = {
            "very coarse": "coarse",  # example if it shows up here too
            "very fine": "fine",
        }
        df["texture_qualifier"] = df["texture_qualifier"].replace(replacements)

        # 3. Handle NA / missing values
        df["texture_qualifier"] = df["texture_qualifier"].replace({"nan": None})

        # 4. Collapse categories (keep small groups)
        valid_categories = {"fine", "coarse",}
        df["texture_qualifier"] = df["texture_qualifier"].where(
            df["texture_qualifier"].isin(valid_categories)
        )
        # 6. Join back
        df = pd.get_dummies(df, columns=["texture_qualifier"], prefix="TEXT_QUAL", dtype=int)
        
        return df

    ## Preprocess texture_value column,
        ## This function cleans and encodes the `texture_value` column into binary features indicating presence of key texture components.
    def preprocess_texture_value(df: pd.DataFrame) -> pd.DataFrame:
        # 1. Normalize strings (lowercase, strip spaces)
        df["texture_value"] = (
            df["texture_value"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        # 2. Replace common typos / variants
        replacements = {
            "stoney": "stony",  # example if it shows up here too
        }
        df["texture_value"] = df["texture_value"].replace(replacements)
        # 3. Handle NA / missing values
        df["texture_value"] = df["texture_value"].replace({"nan": None})
        df["texture_value"] = df["texture_value"].fillna("unknown")

        df["TEXT_sand"] = df["texture_value"].apply(
            lambda x: 1 if any(term in x for term in ["sand", "sandy"]) else 0
        )
        df["TEXT_silt"] = df["texture_value"].apply(
            lambda x: 1 if any(term in x for term in ["silt", "silty"]) else 0
        )
        df["TEXT_clay"] = df["texture_value"].apply(
            lambda x: 1 if any(term in x for term in ["clay"]) else 0
        )
        df["TEXT_loam"] = df["texture_value"].apply(
            lambda x: 1 if any(term in x for term in ["loam", "loamy"]) else 0
        )
        df["TEXT_organic"] = df["texture_value"].apply(
            lambda x: 1 if any(term in x for term in ["peat"]) else 0
        )  
        
        # 6. Join back
        df = df.drop(columns=["texture_value"])    
        return df

    ## Preprocess legacy_texturemodifier column,
        ## This function cleans and encodes the `legacy_texturemodifier` column into binary features indicating presence of key texture modifiers.
        ## This is done in a way to avoid double counting with texture_value features and texture_qualifier
    def preprocess_legacy_texture_mod(df: pd.DataFrame) -> pd.DataFrame:

        # 1. Normalize strings (lowercase, strip spaces)
        df["texture_legacy_texturemodifier"] = (
            df["texture_legacy_texturemodifier"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # 2. Replace common typos / variants
        replacements = {
            "stoney": "stony",  # example if it shows up here too
        }
        df["texture_legacy_texturemodifier"] = df["texture_legacy_texturemodifier"].replace(replacements)

        # 3. Handle NA / missing values
        df["texture_legacy_texturemodifier"] = df["texture_legacy_texturemodifier"].replace({"nan": None})
        df["texture_legacy_texturemodifier"] = df["texture_legacy_texturemodifier"].fillna("unknown")

        df["TEXT_stone"] = df["texture_legacy_texturemodifier"].apply(
            lambda x: 1 if any(term in x for term in ["stony", "bouldery"]) else 0
        )

        df["TEXT_gravel"] = df["texture_legacy_texturemodifier"].apply(
            lambda x: 1 if any(term in x for term in ["gravelly"]) else 0
        )
        
        df["TEXT_loam"] = df.apply(lambda row: 1 if any(term in row["texture_legacy_texturemodifier"] for term in ["loamy"]) else row["TEXT_loam"], axis=1)
        df["TEXT_silt"] = df.apply(lambda x: 1 if any(term in x["texture_legacy_texturemodifier"] for term in ["silty"]) else x["TEXT_silt"], axis=1)
        df["TEXT_QUAL_coarse"] = df.apply(lambda x: 1 if any(term in x["texture_legacy_texturemodifier"] for term in ["coarse"]) else x["TEXT_QUAL_coarse"], axis=1)
        df["TEXT_organic"] = df.apply(lambda x: 1 if any(term in x["texture_legacy_texturemodifier"] for term in ["humic", "peaty"]) else x["TEXT_organic"], axis=1)


        df["TEXT_MOD_gritty"] = df["texture_legacy_texturemodifier"].apply(lambda x: 1 if "gritty" in x else 0)
        df["TEXT_MOD_heavy"] = df["texture_legacy_texturemodifier"].apply(lambda x: 1 if "heavy" in x else 0)
        
        # 6. Join back
        df = df.drop(columns=["texture_legacy_texturemodifier"])    
        return df

    ## Preprocess consistence_strength column,
        ## This function cleans and encodes the `consistence_strength` column into an ordinal numeric value.
        ## this could be binary encoded too, but ordinal is more compact and makes sense here.
    def preprocess_consistence_strength(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and encode the `consistence_strength` column.
        
        Parameters:
        - df: pandas DataFrame with column `consistence_strength`
        - ordinal: if True, adds ordinal encoding; one-hot encoding is always applied
        """
        # 1. Normalize strings
        df["consistence_strength"] = (
            df["consistence_strength"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        
        # 2. Handle missing / unknown
        df["consistence_strength"] = df["consistence_strength"].replace({"nan": None})
        df["consistence_strength"] = df["consistence_strength"].fillna("unknown")
        
        # 5. Optional ordinal encoding
        ordinal_map = {
            "loose": 1,
            "very weak": 2,
            "weak": 3,
            "moderately weak": 4,
            "slightly firm": 5,
            "moderately firm": 6,
            "firm": 7,
            "moderately strong": 8,
            "very firm": 9,
            "very strong": 10,
            "rigid": 11,
            "unknown": 0,
            "other": 0
        }
        df["CONS_strength_val"] = df["consistence_strength"].apply(
            lambda x: ordinal_map.get(x, 0)
        )
        
        return df

    ## Split NZSC soil classifier code into categorical components,
        ## This function splits the NZSC soil classifier code into up to 4 separate categorical columns, to better allow the model to learn patterns.
    def split_nzsc_soil_code(df: pd.DataFrame) -> pd.DataFrame:
        """
        Split a NZSC soil classifier code (1-4 chars) into 4 categorical columns.
        """
        # Extract the 1-4 char code
        df["soil_code"] = df["classifier_nzsc"].str.split().str[0]
        
        
        # Split into characters
        df["NZSC_1"] = df["soil_code"].str[0].fillna("")
        df["NZSC_2"] = df["soil_code"].str[1].fillna("")
        df["NZSC_3"] = df["soil_code"].str[2].fillna("")
        df["NZSC_4"] = df["soil_code"].str[3].fillna("")
        
        # Convert to categorical
        for i in range(1, 5):
            df[f"NZSC_{i}"] = df[f"NZSC_{i}"].astype("category")
        
        # Optional: drop intermediate column
        df = df.drop(columns=["soil_code"])
        
        return df

    ## Preprocess profile_drainage column,
        ## This function cleans and encodes the `profile_drainage` column into an ordinal numeric value.
        ## this could be binary encoded too, but ordinal is more compact and makes sense
    def preprocess_profile_drainage(df: pd.DataFrame) -> pd.DataFrame:

        # 1. Normalize strings
        df["profiledrainage"] = (
            df["profiledrainage"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.rstrip(".")
        )
        
        # 2. Handle missing / unknown
        df["profiledrainage"] = df["profiledrainage"].replace({"nan": None})
        df["profiledrainage"] = df["profiledrainage"].fillna("unknown")
        
        # 5. Optional ordinal encoding
        ordinal_map = {
            "very poor": 1,
            "poor": 2,
            "imperfect": 3,
            "moderately well": 4,
            "well": 5,
            "somewhat excessive": 6,
            "excessive": 7,
            "unknown": 0,
            "other": 0
        }
        df["PROF_drainage"] = df["profiledrainage"].apply(
            lambda x: ordinal_map.get(x, 0)
        )
        
        return df

    ## Preprocess matrix_hue and matrix_value columns,
        ## This function cleans the `matrix_hue` and presents it as a categorical feature, 
        # cleans `matrix_value` and `matrix_chroma` and presents it as the ratio between them.
    def preprocess_matrix_color(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean matrix_hue and matrix_value columns.
        """
        # -------- matrix_hue --------
        # Lowercase, strip spaces
        df["matrix_hue"] = df["matrix_hue"].astype(str).str.upper().str.strip()
        
        # Replace separators with a standard one, e.g., ','
        df["matrix_hue"] = df["matrix_hue"].str.replace(r"[;,/]", ",", regex=True)
        
        # Keep first hue only
        df["matrix_hue"] = df["matrix_hue"].replace({"nan": None})
        df["matrix_hue"] = df["matrix_hue"].fillna("unknown")
        df["CLR_matrix_hue"] = df["matrix_hue"].str.split(",").str[0].fillna("UNK")
        
        # Convert to categorical
        df["CLR_matrix_hue"] = df["CLR_matrix_hue"].astype("category")
        df["CLR_matrix_hue"] = df["CLR_matrix_hue"].replace({"NAN":"UNK"})
                
        # -------- matrix_value --------
            ## moved albedo calcuation to the inital data processing step            
        return df

    ## Preprocess genetic keywords from classifer_nz_genetic column,
        ## This function converts the `classifer_nz_genetic` column into binary keyword features.
        ## This might be destorying some information, but it is very messy data and this should capture the main patterns.
    def preprocess_genetic_keywords(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the classifer_nz_genetic column into binary keyword features.
        
        Adds columns:
            genetic_kw_podzol, genetic_kw_gley, genetic_kw_organic, ...
        """
        # Define keywords
        keywords = [
            "podzol", "gley", "organic", "regosol",
            "grey", "yellow","brown", "pumice", "granular",
            "loam", "sand", "clay", "saline",
            "intergrade", "composite"
        ]
        
        df["classifer_nz_genetic"] = df["classifer_nz_genetic"].replace({"nan": None})
        df["classifer_nz_genetic"] = df["classifer_nz_genetic"].fillna("unknown")
        
        
        # Ensure column is string type
        df["classifer_nz_genetic"] = df["classifer_nz_genetic"].astype(str).str.lower()
        df["classifer_nz_genetic"] = df["classifer_nz_genetic"].str.lower().replace("recent soil", "regosol")
        df["classifer_nz_genetic"] = df["classifer_nz_genetic"].str.lower().replace("-", " ")
        
        
        # Create binary columns
        for kw in keywords:
            col_name = f"GENE_{kw}"
            df[col_name] = df["classifer_nz_genetic"].str.contains(kw).astype(int)
        
        return df

def ConvertNSD_To_ML_readable(data):
    
    # Example dataset
    processed = data.copy()

    parsed = processed["designation"].apply(ML_preprocessor.parse_designation)
    parsed_df = pd.DataFrame(parsed.tolist())

    # explode suffixes into binary flags
    for s in set(sum(parsed_df["DESG_suffixes"], [])):
        parsed_df[f"DESG_suffix_{s}"] = parsed_df["DESG_suffixes"].apply(lambda x: 1 if s in x else 0)

    parsed_df = parsed_df.drop(columns=["DESG_suffixes"])
    final = pd.concat([processed,parsed_df], axis=1)
    processed = final.drop(columns=["designation"])
    processed = processed.drop(columns=["sd_soil_id"])
    
    # processed["horizonnumber"] = processed["horizonnumber"]
    processed["horizondepth_maxval"] = processed["horizondepth_maxval"].astype(float)
    processed["horizondepth_minval"] = processed["horizondepth_minval"].astype(float)
    processed["total_carbon_percent"] = processed["total_carbon_percent"].astype(float)
    processed["moisturefactor"] = processed["moisturefactor"].astype(float)
    processed["cec"] = processed["cec"].astype(float)
    
    processed["clay_percent"] = processed["clay_percent"].astype(float)
    processed["silt_percent"] = processed["silt_percent"].astype(float)
    processed["sand_percent"] = processed["sand_percent"].astype(float)
    processed["rock_fragment_percent"] = processed["rock_fragment_percent"].astype(float)
    processed["whole_bulk_density"] = processed["whole_bulk_density"].astype(float)
    processed["available_water_capacity"] = processed["available_water_capacity"].astype(float)
    
    processed = ML_preprocessor.preprocess_texture_qualifier(processed)

    processed = ML_preprocessor.preprocess_texture_value(processed)
    
    processed = ML_preprocessor.preprocess_legacy_texture_mod(processed)
    
    processed = ML_preprocessor.preprocess_consistence_strength(processed)
    processed = processed.drop(columns=["consistence_strength"])
    
    processed = ML_preprocessor.split_nzsc_soil_code(processed)
    processed = processed.drop(columns=["classifier_nzsc"])
    
    processed = ML_preprocessor.preprocess_profile_drainage(processed)
    processed = processed.drop(columns=["profiledrainage"])
    
    processed = ML_preprocessor.preprocess_matrix_color(processed)
    processed = processed.drop(columns=["matrix_hue", "matrix_value", "matrix_chroma"])
        
    processed = ML_preprocessor.preprocess_genetic_keywords(processed)
    processed = processed.drop(columns=["classifer_nz_genetic"])   
    
    return processed

