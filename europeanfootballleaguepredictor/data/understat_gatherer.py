import numpy as np
import aiohttp
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime as dt
import datetime
from understat import Understat
from loguru import logger
from tqdm import tqdm
import os
import sqlite3
from europeanfootballleaguepredictor.data.database_handler import DatabaseHandler



class Understat_Parser():
    """The class responsible for using the understat api and combining the data with the data_co_uk dataset"""
    
    def __init__(self, league: str, dictionary : dict, database: str) -> None:
        """
        Initializes the understat parser using the league and dictionary for team names used in data_co_uk dataset.

        Args:
            league (str): The league for which the understat parser will request data.
            dictionary (dict): A team name mapping to communicate different team names between data_co_uk and understat.
            database (str): The path to the database of the league.
        """
        self.league = league
        self.dictionary = dictionary
        self.upcoming_fixtures_table = "UpcomingFixtures"
        self.database_handler = DatabaseHandler(database=database, league=league)
        
    def replace_team_names(self, table_names: list, replacing_dict: dict) -> None:
        """
        Replaces the team names used by data_co_uk with the format of understat.

        Args:
            table_names (list): The list of table names to replace team names.
            replacing_dict (dict): A team name mapping to communicate different team names between data_co_uk and understat.
        """
        dataframes_list = self.database_handler.get_data(table_names)
        for dataframe in dataframes_list:
            # Replace team names using the mapping dictionary
            dataframe["HomeTeam"].map(replacing_dict)
            dataframe["AwayTeam"].map(replacing_dict)
        
        self.database_handler.save_dataframes(dataframes=dataframes_list, table_names=table_names)


    async def get_understat_season(self, season: str, months_of_form: int, output_table_name: str) -> None:
        """
        An asynchronous function that connects with the understat api and collects data to construct the required per season datasets.

        Args:
            season (str): The season for which the data gathering process takes place.
            months_of_form (int): The number of months form to take into account when gathering data.
            output_table_name (str): The table name of the database the results will be saved.
        """
        async with aiohttp.ClientSession(cookies={'beget':'begetok'}) as session:
            data_co_uk_table_name = f"DataCoUk_Season{season}_{str(int(season)+1)}"
            self.replace_team_names(table_names = data_co_uk_table_name, replacing_dict=self.dictionary)
            logger.info(f'Started collecting {season} for {self.league} and {months_of_form} month(s) of form.')
            raw_dataframe_list = []
            pd.options.mode.copy_on_write = True
            understat = Understat(session)

            logger.info('Reading the DataCoUk file.')
            #reading the file and keeping only the finished matches
            season_dataframe = self.database_handler.get_data(table_names=data_co_uk_table_name)[0]
            season_dataframe = season_dataframe.dropna(subset=['FTHG'])

            logger.success('Finished reading the file.')
            #creating a new Result column
            season_dataframe['Result'] = np.nan
            for i, x in enumerate(season_dataframe['Result']):
                season_dataframe.loc[i, 'Result'] = f'{season_dataframe.loc[i, "FTHG"]}-{season_dataframe.loc[i, "FTAG"]}'

            #keeping only the important columns and giving appropriate date format
            try:
                season_dataframe = season_dataframe[['Date', 'HomeTeam', 'AwayTeam', 'Result', 'B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5']]
                season_dataframe = season_dataframe.rename(columns={'B365>2.5': 'OverOdds', 'B365<2.5': 'UnderOdds', 'B365H': 'HomeWinOdds', 'B365D': 'DrawOdds', 'B365A': 'AwayWinOdds'})
            except KeyError:
                season_dataframe = season_dataframe[['Date', 'HomeTeam', 'AwayTeam', 'Result', 'B365H', 'B365D', 'B365A', 'BbAv>2.5', 'BbAv<2.5']]
                season_dataframe = season_dataframe.rename(columns={'BbAv>2.5': 'OverOdds', 'BbAv<2.5': 'UnderOdds', 'B365H': 'HomeWinOdds', 'B365D': 'DrawOdds', 'B365A': 'AwayWinOdds'})

            season_dataframe['Date'] = pd.to_datetime(season_dataframe['Date'], format='%d/%m/%Y')
            season_dataframe['Date'] = season_dataframe['Date'].dt.strftime('%d/%m/%Y')

            #get the unique dates for all the matches
            unique_dates = season_dataframe["Date"].unique()

            #loop through the unique dates and gather data from understat from one day prior
            for i, date in tqdm(enumerate(unique_dates), total=len(unique_dates), desc=logger.info('Beggining processing the Understat data.')):
                #formating date and subtracting a day to make sure the match has not been processed by the league table
                unique_dates[i] = datetime.datetime.strptime(unique_dates[i], '%d/%m/%Y').date()
                unique_dates[i] = unique_dates[i] - datetime.timedelta(days=1)

                if months_of_form == None:
                    table = await understat.get_league_table(self.league, season , end_date = str(unique_dates[i]))
                else: 
                    start = unique_dates[i] - relativedelta(months=months_of_form)        
                    table = await understat.get_league_table(self.league, season , end_date = str(unique_dates[i]), start_date = str(start))

                Table = pd.DataFrame(table)
                Table.columns = Table.iloc[0]
                Table = Table[1:]

                Teams = season_dataframe.loc[season_dataframe['Date'] == date]

                #HomeTeam dataframe with respective stats
                HomeStats = Table.loc[Table['Team'].isin(Teams['HomeTeam'].to_list())]
                HomeStats['Team'] = pd.Categorical(
                    HomeStats['Team'], 
                    Teams['HomeTeam'].to_list(), 
                    ordered=True
                    )
                #adding column H prefix
                HomeStats.columns = ['H' + col for col in HomeStats.columns]

                    #AwayTeam dataframe with respective stats
                AwayStats = Table.loc[Table['Team'].isin(Teams['AwayTeam'].to_list())] 
                AwayStats['Team'] = pd.Categorical(
                    AwayStats['Team'], 
                    Teams['AwayTeam'].to_list(), 
                    ordered=True
                    )
                #adding column A prefix
                AwayStats.columns = ['A' + col for col in AwayStats.columns]

                #Combining the season_dataframe with HomeStats and AwayStats
                combined_result_home = pd.merge(Teams, HomeStats, left_on='HomeTeam', right_on='HTeam', how='outer') 
                combined_result_home_away = pd.merge(combined_result_home, AwayStats, left_on='AwayTeam', right_on='ATeam', how='outer')

                raw_dataframe_list.append(combined_result_home_away)

            # Concatenate all DataFrames in the list into a final DataFrame
            final_raw = pd.concat(raw_dataframe_list, ignore_index=True)

            # Save the final DataFrame to the database
            output_table_name = f"{output_table_name}_Season{season}_{str(int(season)+1)}"
            self.database_handler.save_dataframes(dataframes=final_raw, table_names=output_table_name)
    
    async def get_upcoming_match_stats(self, current_season: str, months_of_form_list: int) -> None:
        """
        An asynchronous function that gathers the data for the upcoming matches prediction.

        Args:
            current_season (str): The current season as a string identifier. '2023' represents 2023/2024 season.
            months_of_form_list (int): The number of months form to take into account when gathering data.
        """
        async with aiohttp.ClientSession(cookies={'beget':'begetok'}) as session:
            self.replace_team_names(table_names = self.upcoming_fixtures_table, replacing_dict=self.dictionary)
                    
            logger.info(f'Started collecting upcoming matches statistics')
            pd.options.mode.copy_on_write = True
            understat = Understat(session)
            
            upcoming_matches = self.database_handler.get_data(table_names=self.upcoming_fixtures_table)[0]
            current_date = dt.now()

            dataframes_list = []
            for months_of_form in months_of_form_list:
                if months_of_form == None:
                    table = await understat.get_league_table(self.league, current_season , end_date = str(current_date.strftime("%Y-%m-%d")))
                else: 
                    start = current_date - relativedelta(months=months_of_form)        
                    table = await understat.get_league_table(self.league, current_season , end_date = str(current_date.strftime("%Y-%m-%d")), start_date = str(start.strftime("%Y-%m-%d")))

                Table = pd.DataFrame(table)
                Table.columns = Table.iloc[0]
                Table = Table[1:]


                #HomeTeam dataframe with respective stats
                HomeStats = Table.loc[Table['Team'].isin(upcoming_matches['HomeTeam'].to_list())]
                #adding column H prefix
                HomeStats.columns = ['H' + col for col in HomeStats.columns]

                #AwayTeam dataframe with respective stats
                AwayStats = Table.loc[Table['Team'].isin(upcoming_matches['AwayTeam'].to_list())] 
                #adding column A prefix
                AwayStats.columns = ['A' + col for col in AwayStats.columns]

                #Combining the season_dataframe with HomeStats and AwayStats
                combined_result_home = pd.merge(upcoming_matches, HomeStats, left_on='HomeTeam', right_on='HTeam', how='outer') 
                combined_result_home_away = pd.merge(combined_result_home, AwayStats, left_on='AwayTeam', right_on='ATeam', how='outer')

                dataframes_list.append(combined_result_home_away)
            
            self.database_handler.save_dataframes(table_names=["Raw_UpcomingLongTerm", "Raw_UpcomingShortTerm"], dataframes=dataframes_list)        
            logger.success('Finished understat session for upcoming matches.')