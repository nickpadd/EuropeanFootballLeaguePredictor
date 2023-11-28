import pandas as pd 
from loguru import logger  
import requests
import os
from datetime import datetime, timedelta
from europeanfootballleaguepredictor.data.understat_gatherer import Understat_Parser
import uuid
import tempfile
import asyncio
from europeanfootballleaguepredictor.data.preprocessor import Preprocessor
from europeanfootballleaguepredictor.utils.path_handler import PathHandler
from europeanfootballleaguepredictor.data.database_handler import DatabaseHandler

class UpcomingMatchScheduler():
    """The class responsible for making the important database updates in order to predict the upcoming matches"""
    def __init__(self, current_season: str, months_of_form_list: list, league: str, data_co_uk_ulr: str, data_co_uk_dict: dict, fixtures_url: str, fixtures_dict: dict, database = str, odds : pd.DataFrame =None) ->None:
        """
        Initializing the class

        Args:
            odds (pd.DataFrame, None): A dataframe containing the bookmaker odds for the upcoming matches. Default is None in case the user does not have access to the bookmaker url.
            current_season (str): The current season. '2023' corresponds to 2023/2024 season
            months_of_form_list (list): A list containing the long term form and short term form months. None corresponds to season-long form
            league (str): A string identifier of the league to gather. One of the available ['EPL', 'La_Liga', 'Bundesliga', 'Ligue_1', 'Serie_A']
            data_co_uk_ulr (str): The url that contains the current season dataset download for the corresponding league from the data.co.uk website
            data_co_uk_dict (dict): The dictionary that maps team names of data_co_uk format to understat format
            fixtures_url (str): The url to update the season fixtures 
            fixtures_dict (dict): The dictionary that maps team names of fixtures dataset format to understat format
            database (str): The database name corresponding to the league
        """
        self.odds = odds 
        self.league = league
        self.current_season = current_season 
        self.months_of_form_list = months_of_form_list
        self.data_co_uk_url = data_co_uk_ulr
        self.data_co_uk_dict = data_co_uk_dict
        self.fixtures_url = fixtures_url
        self.fixtures_dict = fixtures_dict
        self.database_handler = DatabaseHandler(league=league, database=database)
        self.upcoming_table = "UpcomingFixtures"
        self.data_co_uk_current_season_table = f"DataCoUk_Season{self.current_season}_{str(int(self.current_season) + 1)}"
        self.fixtures_table = "SeasonFixtures"
    
    def update_dataset(self, category: str) -> None:
        """
        Updates the datasets of the specified category

        Args:
            category (str): A string identifier of what datasets to update. One of available 'odds', 'fixtures'
        """
        if category == 'odds':
            table_name, replacing_dict, url = self.data_co_uk_current_season_table, self.data_co_uk_dict, self.data_co_uk_url
            path = f"/europeanfootballleaguepredictor/data/leagues/{self.league}/DataCoUkFiles/E0-{self.current_season}.csv"
        
        elif category == 'fixtures':
            table_name, replacing_dict, url = self.fixtures_table, self.fixtures_dict, self.fixtures_url
            
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            
            response = requests.get(url)
            if response.status_code == 200:
                temp_path = temp_file.name
                # Write the content to the temporary file
                temp_file.write(response.content)

                # Perform other operations (replace_team_names) on the temporary file
                self.replace_team_names(data_path=temp_path, replacing_dict=replacing_dict)
                
                data = pd.read_csv(temp_path)
                self.database_handler.save_dataframes(dataframes=data, table_names=table_name)
                
                os.remove(temp_path)
                # Log success
                logger.success(f"File downloaded and saved to database table {table_name}")

            else:
                # Log failure
                logger.error(f"Failed to download the file. Status code: {response.status_code}")

            
    def replace_team_names(self, data_path :str, replacing_dict : dict) -> None:
        """
        Replaces the team names using a dictionary mapping in the specified file of the data_path

        Args:
            data_path (str): The path to the file to replace the team names of
            replacing_dict (dict): The dictionary that maps team names of the dataset format to understat format
        """
        logger.info('Replacing team names.')
        data = pd.read_csv(data_path)
        team_dict = replacing_dict
        try:
            data['HomeTeam'] = data['HomeTeam'].replace(team_dict)
            data['AwayTeam'] = data['AwayTeam'].replace(team_dict)
        except KeyError:
            data['Home Team'] = data['Home Team'].replace(team_dict)
            data['Away Team'] = data['Away Team'].replace(team_dict)
            
        data.to_csv(data_path, index=False)
        
    def setup_upcoming_fixtures(self) ->None:
        """A pipeline to read update and write the updated datasets in order for the model to predict upcoming matches"""
        fixtures = self.database_handler.get_data(table_names=self.fixtures_table)[0]
        fixtures.rename(columns={'Home Team': 'HomeTeam', 'Away Team': 'AwayTeam'}, inplace=True)
        fixtures['Date'] = pd.to_datetime(fixtures['Date'], format="%d/%m/%Y %H:%M")
        fixtures['Date'] = fixtures['Date'].dt.strftime("%d/%m/%Y")
        
        try:
            upcoming_matches = pd.merge(fixtures[['Date', 'HomeTeam', 'AwayTeam']], self.odds, left_on=['HomeTeam', 'AwayTeam'], right_on=['Home Team', 'Away Team'], how='inner')
            upcoming_matches = upcoming_matches.drop(columns=['Home Team', 'Away Team'], axis=1)
            upcoming_matches.rename(columns={'1': 'HomeWinOdds', 'x': 'DrawOdds', '2': 'AwayWinOdds', 'OverLine': 'OverLineOdds', 'UnderLine': 'UnderLineOdds'}, inplace=True)
            upcoming_matches.drop(columns=['Yes', 'No'])
        except (KeyError, TypeError) as e:
            today = datetime.now()
            fifteen_days_from_now = today + timedelta(days=15)
            fixtures['Date'] = pd.to_datetime(fixtures['Date'], format='%d/%m/%Y')
            upcoming_matches = fixtures[(fixtures['Date'] >= today) & (fixtures['Date'] <= fifteen_days_from_now)][['Date', 'HomeTeam', 'AwayTeam']]
            upcoming_matches['Date'] = upcoming_matches['Date'].dt.strftime('%d/%m/%Y')
             
        self.database_handler.save_dataframes(dataframes=upcoming_matches, table_names=self.upcoming_table)
        logger.success(f'Upcoming fixtures saved succesfully at {self.upcoming_table}')
        
        database = f'europeanfootballleaguepredictor/data/database/{self.league}_database.db'
        understat_parser = Understat_Parser(league=self.league, dictionary=self.data_co_uk_dict, database=database)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(understat_parser.get_upcoming_match_stats(current_season= self.current_season, months_of_form_list= self.months_of_form_list))
            
        preprocessor = Preprocessor(league=self.league, database=database)
        upcoming_raw_long, upcoming_raw_short = preprocessor.database_handler.get_data(table_names=["Raw_UpcomingLongTerm", "Raw_UpcomingShortTerm"])
        upcoming_preprocessed_list = preprocessor.preprocessing_pipeline(data = [upcoming_raw_long, upcoming_raw_short])
        preprocessor.database_handler.save_dataframes(dataframes=upcoming_preprocessed_list, table_names=["Preprocessed_UpcomingLongTerm", "Preprocessed_UpcomingShortTerm"])
        
        
