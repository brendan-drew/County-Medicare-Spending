import pandas as pd
import numpy as np
import os

'''
THIS FILE IS USED TO LOAD AND PROCESS DATA PRIOR TO USE IN THE ANALYSIS
INCLUDES CODE TO UPLOAD DATAFRAMES FOR THE FOLLOWING INFORMATION:
- Medicare spending data from the Dartmouth Institute from 2003 - 2015
  source: http://www.dartmouthatlas.org/tools/downloads.aspx#spending
- Primary care access/utilization data from the Dartmouth Institute from 2003 - 2015
  source: http://www.dartmouthatlas.org/tools/downloads.aspx#spending
- 2014 ACO performance data from CMS
  source: https://data.cms.gov/ACO/Medicare-Shared-Savings-Program-Accountable-Care-O/ucce-hhpu
- Detailed beneficiary demographic/utilization data and spending data by category from CMS for 2003 - 2015
  source: https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Geographic-Variation/GV_PUF.html
'''

# Loads Medicare spending by county from the Dartmouth Institute
def load_medicare_spending():
    '''
    INPUT: None
    OUTPUT: DataFrame with Medicare Spending Data from the Dartmouth Institute from 2003-2015
    '''
    years = range(2003, 2015)
    df_list = []
    for yr in years:
        file_name = 'pa_reimb_county_' + str(yr) + '.xls'
        file_path = os.path.join('data', 'Medicare Spending by County', file_name)
        df = pd.read_excel(file_path, skip_rows = range(2), names = ['county_id', 'county_name', 'medicare_enrollees', 'total_reimb', 'total_reimb_price_adj', 'hospital_snf_reimb', 'hospital_snf_reimb_price_adj', 'physician_reimb', 'physician_reimb_price_adj', 'outpatient_reimb', 'outpatient_reimb_price_adj', 'home_health_reimb', 'home_health_reimb_price_adj', 'hospice_reimb', 'hospice_reimb_price_adj', 'dme_reimb', 'dme_reimb_price_adj'])
        df.dropna(axis = 0, subset = ['county_id'], inplace = True)
        df.dropna(axis = 0, thresh = 14, inplace = True)
        df['year'] = yr
        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col])
        df['state'] = [name[:2] if isinstance(name, unicode) else 'unknown' for name in df['county_name']]
        df['county_excl_state'] = [name[3:] if isinstance(name, unicode) else 'unknown' for name in df['county_name']]
        df_list.append(df)
    output_df = pd.concat(df_list)
    return output_df

# Loads the primary care access data from the Dartmouth Institute
def load_pc_access():
    '''
    INPUT: None
    OUTPUT: DataFrame with primary care access data from the Dartmouth Institute from 2003 - 2015 (Note that the data from 2003 - 2007 is aggregated)
    '''
    years = ['0307'] + range(2008,2015)
    df_list = []
    for yr in years:
        file_name = 'PC_County_rates_' + str(yr) + '.xls'
        file_path = os.path.join('data', 'county_primary_care_access', file_name)
        df = pd.read_excel(file_path, skip_rows = range(3), names = ['county_id', 'county_name', 'medicare_enrollees', 'black_medicare_enrollees', 'white_medicare_enrollees', 'rate_one_ambulatory_visit', 'lower_ci_one_amb_visit', 'upper_ci_one_amb_visit', 'rate_one_ambulatory_visit_black', 'lower_ci_one_amb_visit_black', 'upper_ci_one_amb_visit_black', 'rate_one_ambulatory_visit_white', 'lower_ci_one_amb_visit_white', 'upper_ci_one_amb_visit_white', 'diabetic_enrollees', 'black_diabetic_enrollees', 'white_diabetic_enrollees', 'diabetic_pts_a1c', 'lower_ci_diabetic_pts_a1c', 'upper_ci_diabetic_pts_a1c', 'diabetic_pts_a1c_black', 'lower_ci_diabetic_pts_a1c_black', 'upper_ci_diabetic_pts_a1c_black', 'diabetic_pts_a1c_white', 'lower_ci_diabetic_pts_a1c_white', 'upper_ci_diabetic_pts_a1c_white', 'eye_exam', 'lower_ci_eye_exam', 'upper_ci_eye_exam', 'eye_exam_black', 'lower_ci_eye_exam_black', 'upper_ci_eye_exam_black', 'eye_exam_white', 'lower_ci_eye_exam_white', 'upper_ci_eye_exam_white', 'diabetic_pts_lipids', 'lower_ci_diabetic_pts_lipids', 'upper_ci_diabetic_pts_lipids', 'diabetic_pts_lipids_black', 'lower_ci_diabetic_pts_lipids_black', 'upper_ci_diabetic_pts_lipids_black', 'diabetic_pts_lipids_white', 'lower_ci_diabetic_pts_lipids_white', 'upper_ci_diabetic_pts_lipids_white', 'female_enrollees', 'female_enrollees_black', 'female_enrollees_white', 'mammogram', 'lower_ci_mammogram', 'upper_ci_mammogram', 'mammogram_black', 'lower_ci_mammogram_black', 'upper_ci_mammogram_black', 'mammogram_white', 'lower_ci_mammogram_white', 'upper_ci_mammogram_white', 'beneficiaries_100_part_A', 'beneficiaries_100_part_A_black', 'beneficiaries_100_part_A_white', 'leg_amputations', 'lower_ci_leg_amputations', 'upper_ci_leg_amputations', 'leg_amputations_black', 'lower_ci_leg_amputations_black', 'upper_ci_leg_amputations_black', 'leg_amputations_white', 'lower_ci_leg_amputations_white', 'upper_ci_leg_amputations_white', 'discharges_amb_sensitive', 'lower_ci_discharges_amb_sensitive', 'upper_ci_discharges_amb_sensitive', 'discharges_amb_sensitive_black', 'lower_ci_discharges_amb_sensitive_black', 'upper_ci_discharges_amb_sensitive_black', 'discharges_amb_sensitive_white', 'lower_ci_discharges_amb_sensitive_white', 'upper_ci_discharges_amb_sensitive_white'])
        df['year'] = yr
        df.dropna(axis = 0, subset = ['county_id'], inplace = True)
        df_list.append(df)
    output_df = pd.concat(df_list)
    return output_df

    # Load the ACO performance data
    def load_aco_results():
        '''
        INPUT: None
        OUTPUT: DataFrame with 2014 ACO performance data from Medicare
        '''
        file_path = os.path.join('data', 'aco_results', 'Medicare_Shared_Savings_Program_Accountable_Care_Organizations_Performance_Year_2014_Results.csv')
        df = pd.read_csv(file_path)
        return df

########################## Loading/cleaning Medicare's county-level spending data########################################################################

# Loads data from Medicare's data on county-level health spending, this data was exported used to export to a csv to facilitate data cleaning
def load_medicare_county():
    '''
    INPUT: None
    OUTPUT: Writes data from an Excel file with detailed Medicare benificiary demographic/utilization and spending data by year to a csv file to facilitate data processing
    Note: This function takes a LONG time to run
    '''
    file_path = os.path.join('data', 'medicare_county_level', 'County_All_Table.xlsx')
    years = range(2007, 2015)
    df_list = []
    for yr in years:
        sheetname = 'State_county ' + str(yr)
        df = pd.read_excel(file_path, sheetname = sheetname, header = 1, skip_rows = 0)
        df['year'] = yr
        df_list.append(df)
    output_df = pd.concat(df_list)
    output_df.to_csv(os.path.join('data', 'medicare_county_level', 'medicare_county_all.csv'))
    return

# Clean Medicare's county-level data from the csv file created with load_medicare_county
def clean_medicare_county_csv():
    '''
    INPUT: None
    OUTPUT: Reads in the csv file 'medicare_county_all.csv' (output of the load_medicare_county function) and processes the data for further use. Writes data to a new csv file 'cleaned_medicare_county_all.csv'. The cleaned file should be used for future analysis
    '''
    file_path = os.path.join('data', 'medicare_county_level', 'medicare_county_all.csv')
    df = pd.read_csv(file_path)
    df.columns = [col.lower().replace(' ','_') for col in df.columns]
    df.replace(['.','*'], np.nan, inplace = True)
    # Drop lines missing the county id (these are aggregate state-level columns)
    df.dropna(axis = 0, subset = ['state_and_county_fips_code'], inplace = True)
    # Drop columns with all missing values (these are reported only at the state level)
    for col in df.columns:
        if pd.isnull(df[col]).sum() == df.shape[0]:
            df.drop(col, axis = 1, inplace = True)
    # Write to new csv file
    df.to_csv(os.path.join('data', 'medicare_county_level', 'cleaned_medicare_county_all.csv'))
    return

###############################################################################

if __name__ == '__main__':
    pass
