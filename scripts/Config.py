# columns.py

# Columns to drop from masterdata
raw_master_data_drop = [
      'grids_6k', 'grids_4k','OSTWERT', 'NORDWERT',
      'PERMEABILI', 'RADON_RN_2','Geo_class',
      'RADON_RN_3', 'GRP_conc', 'grids_10pa', 'Carbononif',
      'Dev_low', 'Dev_mid', 'Dev_carb_p','Dev_up', 'Holocene',
      'Palaeo_met', 'Paleo_igne', 'Perm_low_m',
      'Perm_up_Ze', 'Pleistocen', 'Tert_fluv', 'Tert_volc', 'Trias_bund',
      'Trias_musc', 'Trias_keup', 'geometry'
      ]
      
      
# Columns to drop for df_corr
df_corr_columns_to_drop = [
    'grids_36k', 'Carbononif', 'Dev_low', 'Dev_mid', 'Dev_carb_p',
    'Dev_up', 'Holocene', 'Palaeo_met', 'Paleo_igne', 'Perm_low_m',
    'Perm_up_Ze', 'Pleistocen', 'Tert_fluv', 'Tert_volc', 'Trias_bund',
    'Trias_musc', 'Trias_keup'
]

# Columns for df_corr_Geo
df_corr_Geo_columns = [
    'GRP', 'Carbononif', 'Dev_low', 'Dev_mid', 'Dev_carb_p',
    'Dev_up', 'Holocene', 'Palaeo_met', 'Paleo_igne', 'Perm_low_m',
    'Perm_up_Ze', 'Pleistocen', 'Tert_fluv', 'Tert_volc', 'Trias_bund',
    'Trias_musc', 'Trias_keup'
]

# Seasonal variables
seasonal_variables = [
    'Prec_2022_', 'Prec_JJA_2', 'Prec_JJA_3', 'Prec_MAM_2', 'Prec_MAM_3',
    'Prec_SON_2', 'Prec_SON_3', 'Precip_202', 'Precip_203', 'Precip_204',
    'Precip_205', 'Soil_Tem_1', 'Soil_Tem_3', 'SoilTemp_2'
]

# Additional columns to drop
columns_to_drop = [
    'OSTWERT', 'NORDWERT', 'PERMEABILI', 'RADON_RN_2', 'RADON_RN_3',
    'GRP_conc', 'Geo_class', 'TWI_Hessen', 'geometry', 'grids_6k',
    'grids_4k', 'grids_10pa', 'grids_30k', 'grids_40k'
]


feature_groups_list=['Carbononif','Dev_low', 'Dev_mid', 'Dev_carb_p', 'Dev_up',
                'Holocene', 'Palaeo_met','Paleo_igne', 'Perm_low_m', 'Perm_up_Ze',
                'Pleistocen', 'Tert_fluv','Tert_volc', 'Trias_bund', 'Trias_musc',
                'Trias_keup'
                ]


      
rf_predictor_list = [
    'Lin_Densit','Pp_Silt_He','CP_PhH20_H','HC_WP_Ger_','CP_CEC_Hes', 'DEM',
    'Pp_Sand_He','CP_N_Hesse', 'HC_FC_Weig','Carbononif', 'Dev_low', 'Dev_mid',
    'Dev_carb_p','Dev_up', 'Holocene', 'Palaeo_met', 'Paleo_igne', 'Perm_low_m',
    'Perm_up_Ze', 'Pleistocen', 'Tert_fluv', 'Tert_volc', 'Trias_bund',
    'Trias_musc', 'Trias_keup', 'GRP','grids_36k'
    ]

svr_predictor_list = ['CP_N_Hesse', 'CP_P_Hesse','Pp_CoarseP','CP_CaC03_H','HC_WP_Ger_',
'Pp_Clay_He','CP_CN_Hess','AnnPreci_1', 'Pp_Sand_He','Carbononif', 'Dev_low', 'Dev_mid',
    'Dev_carb_p','Dev_up', 'Holocene', 'Palaeo_met', 'Paleo_igne', 'Perm_low_m',
    'Perm_up_Ze', 'Pleistocen', 'Tert_fluv', 'Tert_volc', 'Trias_bund',
    'Trias_musc', 'Trias_keup', 'GRP','grids_36k']

gbr_predictor_list = ['CP_N_Hesse', 'CP_P_Hesse','Pp_CoarseP','CP_CaC03_H','HC_WP_Ger_',
'Pp_Clay_He','CP_CN_Hess','HC_FC_Weig', 'Pp_Sand_He','Carbononif', 'Dev_low', 'Dev_mid',
    'Dev_carb_p','Dev_up', 'Holocene', 'Palaeo_met', 'Paleo_igne', 'Perm_low_m',
    'Perm_up_Ze', 'Pleistocen', 'Tert_fluv', 'Tert_volc', 'Trias_bund',
    'Trias_musc', 'Trias_keup', 'GRP','grids_36k']

rename_preds_dict = {
    'Rn_Poten': 'GRP',
    'AnnPreci_1': 'Precp_LTA',
    'CP_CaC03_H': 'CaC03',
    'CP_CEC_Hes': 'CEC',
    'CP_CN_Hess': 'CN',
    'CP_N_Hesse': 'N',
    'CP_P_Hesse': 'P',
    'CP_PhH20_H': 'PhH20',
    'DEM': 'DEM',
    'HC_FC_Weig': 'FC',
    'HC_KS_Weig': 'KS',
    'HC_SWC_Wei': 'SWC',
    'HC_WP_Ger_': 'WP',
    'Lin_Densit': 'Lin_Dens',
    'Pp_Clay_He': 'Clay_fra',
    'Pp_CoarseP': 'Coarse_fra',
    'Pp_Sand_He': 'Sand_fra',
    'Pp_Silt_He': 'Silt_fra',
    'Prec_2020_': 'Precp_2020',
    'Prec_2022_': 'Prec_2022',
    'Prec_JJA_2': 'Prec_JJA_2',
    'Prec_JJA_3': 'Precp_JJA_3',
    'Prec_MAM_2': 'Precp_MAM_2',
    'Prec_MAM_3': 'Precp_MAM_3',
    'Prec_SON_2': 'Prec_SON_2',
    'Prec_SON_3': 'Prec_SON_3',
    'Precip_202': 'Precp_202',
    'Precip_203': 'Precp_203',
    'Precip_204': 'Precp_204',
    'Precip_205': 'Precp_205',
    'Soil_Tem_1': 'Soil_Tem_2020',
    'Soil_Tem_3': 'Soil_Tem_2022',
    'SoilMois_9': 'SoilMois_LTA',
    'SoilTemp_2': 'SoilTemp_LTA',
    'TempAir_19': 'TempAir_19',
    'TWI_Hessen': 'TWI',
    'Uranium_ge': 'Uranium'
}


class RenameUnit:
    def __init__(self, geo_data, default_col):
        """
        Initialize the GeologyDataProcessor with the base GeoDataFrame.
        """
        self.geo_data = geo_data
        self.default_col = default_col

    def replace_vals(self, new_column, ref_column, old_values, new_values):
        """
        Replace values in the GeoDataFrame based on specified mappings.
        
        :param new_column: Column to create with new values.
        :param ref_column: Column to replace values based on.
        :param old_values: List of old values to be replaced.
        :param new_values: List of new values for replacement.
        :return: Updated GeoDataFrame.
        """
        value_mapping = dict(zip(old_values, new_values))
        self.geo_data[new_column] = self.geo_data[ref_column].replace(value_mapping)
        return self.geo_data

    def process_geology_columns(self):
        """
        Applies sequential replacement mappings to create updated geology columns.
        
        :return: Updated GeoDataFrame with new geology columns.
        """
        # Mapping dictionaries
        old_values1 = ['Triassic, Middle (Muschelkalk)','Triassic, Lower (Buntsandstein)',
                       'Triassic, Middle to Upper (Keuper)', 'Holocene', 'Tertiary (fluviatile)',
                       'Tertiary (volcanites)', 'Pleistocene', 'Permian, Upper (Zechstein)',
                       'Carboniferous', 'Jurassic, Lower', 'Paleozoic igneous rocks',
                       'Devonian, Middle', 'Devonian, Upper', 'Silurian', 'Devonian, Lower',
                       'Ordovician', 'Permian, Lower to Middle (Rotliegend)', 'Paleozoic (metamorph)',
                       'Devonian to Carboniferous (plutonites)']
        
        new_values1 = ['Triassic, Middle (Muschelkalk)','Triassic, Lower (Buntsandstein)',
                       'Triassic, Middle to Upper (Keuper)', 'Holocene', 'Tertiary (fluviatile)',
                       'Tertiary (volcanites)', 'Pleistocene', 'Permian, Upper (Zechstein)',
                       'Carboniferous', 'Triassic, Lower (Buntsandstein)', 'Paleozoic igneous rocks',
                       'Devonian, Middle', 'Devonian, Upper', 'Triassic, Lower (Buntsandstein)',
                       'Devonian, Lower', 'Triassic, Lower (Buntsandstein)', 'Permian, Lower to Middle (Rotliegend)',
                       'Paleozoic (metamorph)', 'Devonian to Carboniferous (plutonites)']
        
        old_geo = ['Triassic, Middle (Muschelkalk)', 'Triassic, Lower (Buntsandstein)',
                   'Triassic, Middle to Upper (Keuper)', 'Holocene', 'Tertiary (fluviatile)',
                   'Tertiary (volcanites)', 'Pleistocene', 'Permian, Upper (Zechstein)',
                   'Carboniferous', 'Jurassic, Lower', 'Paleozoic igneous rocks', 'Devonian, Middle',
                   'Devonian, Upper', 'Silurian', 'Devonian, Lower', 'Ordovician',
                   'Permian, Lower to Middle (Rotliegend)', 'Paleozoic (metamorph)',
                   'Devonian to Carboniferous (plutonites)']
        
        new_geo = ['Triassic Muschelkalk', 'Triassic Bunter sandstone', 'Triassic Keuper',
                   'Holocene sedimentary rocks', 'Tertiary fluviatile', 'Tertiary volcanics',
                   'Pleistocene sediments', 'Late Permian Zechstein', 'Carboniferous felsic volcanics',
                   'Triassic Bunter sandstone', 'Paleozoic mafic volcanics', 'Mid-upper Devonian sediments',
                   'Devonian volcanics', 'Triassic Bunter sandstone', 'Devonian fluviatile',
                   'Triassic Bunter sandstone', 'Permian Rotliegend sediments',
                   'Paleozoic Intermediate-felsic plutonites', 'Devonian to Carboniferous plutonites']
        
        new_geoid = ['GEOID 5', 'GEOID 7', 'GEOID 6', 'GEOID 1', 'GEOID 3', 'GEOID 11', 'GEOID 2',
                     'GEOID 4', 'GEOID 12', 'GEOID 7', 'GEOID 14', 'GEOID 8', 'GEOID 13', 'GEOID 7',
                     'GEOID 9', 'GEOID 7', 'GEOID 10', 'GEOID 15', 'GEOID 16']
        
        # Apply replacements
        self.replace_vals('Geo_class', self.default_col, old_values1, new_values1)
        self.replace_vals('class_geo', 'Geo_class', old_geo, new_geo)
        self.replace_vals('GEOID', 'class_geo', new_geo, new_geoid)
        
        return self.geo_data