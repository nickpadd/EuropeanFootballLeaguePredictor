import pandas as pd 
from loguru import logger 
import uuid
import os 
import sys

class Preprocessor():
    def __init__(self, preprocessed_data_path):
        self.preprocessed_data_path = preprocessed_data_path
    
    def get_files(self, folder_path) -> pd.DataFrame:
        '''
        Method that gets all the available .csv files from path.
        Args:
        path: The path of the folder of the .csv files
        Returns:
        combined_df: A dataframe containing the combined seasons of the folder
        '''
        logger.info(f'Parsing the files of {folder_path}')
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        # Initialize an empty DataFrame to store the combined data
        combined_df = pd.DataFrame()

        # Loop through each CSV file, read it, and concatenate it to the combined DataFrame
        for csv_file in csv_files:
            print(csv_file)
            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        logger.success(f'Succesfully parsed the files of {folder_path}')
        return combined_df
    
    def produce_match_id(self, dataframe_list : list = None, dataframe : pd.DataFrame = None):

        dataframes_with_id = []

        for dataframe in dataframe_list:
            dataframe['Match_id'] = dataframe.apply(lambda row: uuid.uuid5(uuid.NAMESPACE_DNS, f"{row['HomeTeam']}_{row['AwayTeam']}_{row['Date']}"), axis=1)
            dataframe = dataframe.drop(columns=['HTeam', 'ATeam'])
            dataframes_with_id.append(dataframe)
        logger.success(f'Succesfully produced Match_id')
        return dataframes_with_id

    def stats_per_match(self, dataframe_list : list = None, dataframe : pd.DataFrame = None):
        
        data_per_match = []
        for dataframe in dataframe_list:
            for col in ['HW', 'HD', 'HL', 'HG', 'HGA', 'HPTS', 'HxG', 'HNPxG', 'HxGA', 'HNPxGA', 'HNPxGD', 'HDC', 'HODC', 'HxPTS']:
                dataframe[col] = dataframe[col].div(dataframe['HM']).fillna(0)
                dataframe.rename(columns = {col: f'{col}/M'}, inplace=True)

            for col in ['AW', 'AD', 'AL', 'AG', 'AGA', 'APTS', 'AxG', 'ANPxG', 'AxGA', 'ANPxGA', 'ANPxGD', 'ADC', 'AODC', 'AxPTS']:
                dataframe[col] = dataframe[col].div(dataframe['AM']).fillna(0)
                dataframe.rename(columns = {col: f'{col}/M'}, inplace=True)

            data_per_match.append(dataframe)
                
        logger.success('Succesfully normalized stats per match.')   
        return data_per_match
                
    def test_for_nulls(self, data, data_status: str):
        filtered_data_list=[]
        odds_columns = ['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverOdds', 'UnderOdds']
        if data_status == 'Raw':
            stats_columns = ['HW', 'HD', 'HL', 'HG', 'HGA', 'HPTS', 'HxG', 'HNPxG', 'HxGA', 'HNPxGA', 'HNPxGD', 'HDC', 'HODC', 'HxPTS', 'HPPDA', 'HOPPDA',
                             'AW', 'AD', 'AL', 'AG', 'AGA', 'APTS', 'AxG', 'ANPxG', 'AxGA', 'ANPxGA', 'ANPxGD', 'ADC', 'AODC', 'AxPTS', 'APPDA', 'AOPPDA']
        if data_status == 'Preprocessed':
            stats_columns = ['HW/M', 'HD/M', 'HL/M', 'HG/M', 'HGA/M', 'HPTS/M', 'HxG/M', 'HNPxG/M', 'HxGA/M', 'HNPxGA/M', 'HNPxGD/M', 'HDC/M', 'HODC/M', 'HxPTS/M', 'HPPDA', 'HOPPDA',
                             'AW/M', 'AD/M', 'AL/M', 'AG/M', 'AGA/M', 'APTS/M', 'AxG/M', 'ANPxG/M', 'AxGA/M', 'ANPxGA/M', 'ANPxGD/M', 'ADC/M', 'AODC/M', 'AxPTS/M', 'APPDA', 'AOPPDA']
        info_columns = ['HomeTeam', 'AwayTeam', 'Result']
        for dataframe in data:
            contains_nulls = dataframe.isnull().any().any()
            if contains_nulls:
                info_has_nulls = dataframe[info_columns].isnull().any()
                stats_have_nulls = dataframe[stats_columns].isnull().any()
                odds_have_nulls = dataframe[odds_columns].isnull().any()
                if info_has_nulls.any():
                    null_rows = dataframe[info_columns].isnull().any(axis=1)
                    logger.error(f'{data_status} data contain NaN values in the team names:\n {dataframe.loc[null_rows, info_columns]} \n Usually, this error occurs when league dictionaries are not updated correctly! \n Ending the run...')
                    sys.exit(1)
                elif stats_have_nulls.any():
                    null_rows = dataframe[stats_columns].isnull().any(axis=1)
                    logger.warning(f'{data_status} data contain NaN values in the statistics:\n {dataframe.loc[null_rows, stats_columns]} \n Usually, this warning occurs due to data_co_uk datasets containing {None} values. DELETING THE ABOVE ENTRIES!')
                    dataframe.dropna(subset=stats_columns, inplace=True)
                    filtered_data_list.append(dataframe)
                elif odds_have_nulls.any():
                    null_rows = dataframe[odds_columns].isnull().any(axis=1)
                    logger.warning(f'{data_status} data contain NaN values in the odds:\n {dataframe.loc[null_rows, odds_columns]} \n Usually, this warning occurs due to data_co_uk datasets containing {None} values. DELETING THE ABOVE ENTRIES!')
                    dataframe.dropna(subset=odds_columns, inplace=True)
                    filtered_data_list.append(dataframe)
                    
            if contains_nulls and data_status=='Preprocessed':
                logger.error('Unexpected nulls in the preprocessed datasets! \n Ending the run...')
                sys.exit(1)
            else:
                logger.success(f'No NaN values were detected!')
                filtered_data_list.append(dataframe)
                
        return filtered_data_list
                
            
    def preprocessing_pipeline(self, data: list):
        data = self.test_for_nulls(data,  data_status='Raw')
        data = self.produce_match_id(data)
        data = self.stats_per_match(data)
        self.test_for_nulls(data, data_status='Preprocessed')
        
        return data
    
    def output_files(self, dataframe_list: list):
        for dataframe, name in zip(dataframe_list, ['LongTermForm.csv', 'ShortTermForm.csv']):
            final_path = os.path.join(self.preprocessed_data_path, name)
            dataframe.to_csv(final_path)
            logger.success(f'Successfully saved preprocessed files to {self.preprocessed_data_path}')
    