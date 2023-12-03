import pandas as pd 
from loguru import logger 
import uuid
import os 
import sys
from europeanfootballleaguepredictor.data.database_handler import DatabaseHandler

class Preprocessor():
    """A class responsible for preprocessing the raw collected datasets
    """
    def __init__(self, league :str, database :str):
        """
        Initializes the Preprocessor.

        Args:
            league (str): The name of the league associated with the preprocessing.
            database (str): The path to the SQLite database file.
        """
        self.database_handler = DatabaseHandler(league = league, database= database)
    
    def produce_match_id(self, dataframe_list : list) -> list:
        """
        Produces a unique identification in 'Match_id' for each match in the list of dataframes.

        Args:
            dataframe_list (list): List of dataframes for which 'Match_id' will be created.

        Returns:
            list: The input list of dataframes with the unique identifier 'Match_id' column.
        """

        dataframes_with_id = []

        for dataframe in dataframe_list:
            dataframe['Match_id'] = dataframe.apply(lambda row: uuid.uuid5(uuid.NAMESPACE_DNS, f"{row['HomeTeam']}_{row['AwayTeam']}_{row['Date']}"), axis=1)
            dataframe = dataframe.drop(columns=['HTeam', 'ATeam'])
            dataframe['Match_id'] = dataframe['Match_id'].astype(str)
            dataframes_with_id.append(dataframe)
        logger.success(f'Succesfully produced Match_id')
        return dataframes_with_id

    def stats_per_match(self, dataframe_list : list) -> list:
        """
        Normalizes the advanced statistics to convey statistics per match.

        Args:
            dataframe_list (list): List of dataframes.

        Returns:
            list or pd.DataFrame: The input list of dataframes with the statistics normalized per match and the columns renamed.
        """
        data_per_match = []
        for dataframe in dataframe_list:
            for col in ['HW', 'HD', 'HL', 'HG', 'HGA', 'HPTS', 'HxG', 'HNPxG', 'HxGA', 'HNPxGA', 'HNPxGD', 'HDC', 'HODC', 'HxPTS']:
                dataframe[col] = dataframe[col].astype(float).div(dataframe['HM'].astype(int)).fillna(0)
                dataframe.rename(columns = {col: f'{col}/M'}, inplace=True)

            for col in ['AW', 'AD', 'AL', 'AG', 'AGA', 'APTS', 'AxG', 'ANPxG', 'AxGA', 'ANPxGA', 'ANPxGD', 'ADC', 'AODC', 'AxPTS']:
                dataframe[col] = dataframe[col].astype(float).div(dataframe['AM'].astype(int)).fillna(0)
                dataframe.rename(columns = {col: f'{col}/M'}, inplace=True)

            data_per_match.append(dataframe)
                
        logger.success('Succesfully normalized stats per match.')   
        return data_per_match
                
    def test_for_nulls(self, data: list, data_status: str) -> list:
        """
        Tests the datasets and deals with null values.

        Args:
            data (list): A list of dataframes or a dataframe to be checked.
            data_status (str): Identifier for the status of the data. One of ['Raw', 'Preprocessed'].

        Returns:
            list: A list of dataframes with checked and dealt with null values according to the data status.
        """
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
                try:
                    info_has_nulls = dataframe[info_columns].isnull().any()
                except KeyError:
                    logger.debug(dataframe.columns)
                    logger.debug(dataframe.head(20))
                stats_have_nulls = dataframe[stats_columns].isnull().any()
                odds_have_nulls = dataframe[odds_columns].isnull().any()
                if info_has_nulls.any():
                    null_rows = dataframe[info_columns].isnull().any(axis=1)
                    logger.error(f'{data_status} data contain NaN values in the team names:\n {dataframe.loc[null_rows, info_columns]} \n Usually, this error occurs when league dictionaries are not updated correctly! \n Ending the run...')
                    sys.exit(1)
                if stats_have_nulls.any():
                    null_rows = dataframe[stats_columns].isnull().any(axis=1)
                    logger.warning(f'{data_status} data contain NaN values in the statistics:\n {dataframe.loc[null_rows, stats_columns]} \n Usually, this warning occurs due to data_co_uk datasets containing {None} values. DELETING THE ABOVE ENTRIES!')
                    logger.warning(f"This might be the case because of the replacing dictionary missing a team. Team names: {dataframe.loc[null_rows, info_columns]}")
                    dataframe.dropna(subset=stats_columns, inplace=True)
                if odds_have_nulls.any():
                    null_rows = dataframe[odds_columns].isnull().any(axis=1)
                    logger.warning(f'{data_status} data contain NaN values in the odds:\n {dataframe.loc[null_rows, odds_columns]} \n Usually, this warning occurs due to data_co_uk datasets containing {None} values. DELETING THE ABOVE ENTRIES!')
                    dataframe.dropna(subset=odds_columns, inplace=True)
                
                filtered_data_list.append(dataframe)
            if contains_nulls and data_status=='Preprocessed':
                logger.error('Unexpected nulls in the preprocessed datasets! \n Ending the run...')
                sys.exit(1)
            if not contains_nulls:
                logger.success(f'No NaN values were detected!')
                filtered_data_list.append(dataframe)
                
        return filtered_data_list
                
            
    def preprocessing_pipeline(self, data: list) -> list:
        """
        A pipeline that produces preprocessed dataframes out of the raw dataframes list.

        Args:
            data (list): A list containing dataframes in the raw format of collected dataframes.

        Returns:
            list: A list of preprocessed dataframes.
        """
        data = self.test_for_nulls(data,  data_status='Raw')
        data = self.produce_match_id(data)
        data = self.stats_per_match(data)
        self.test_for_nulls(data, data_status='Preprocessed')
        
        return data
    
    