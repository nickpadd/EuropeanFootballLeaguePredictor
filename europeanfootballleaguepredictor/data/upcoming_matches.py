import pandas as pd 
from loguru import logger  
import requests
import os
from datetime import datetime
from europeanfootballleaguepredictor.data.understat_gatherer import Understat_Parser
import uuid
import asyncio
from europeanfootballleaguepredictor.data.preprocessor import Preprocessor
from europeanfootballleaguepredictor.utils.path_handler import PathHandler

class UpcomingMatchScheduler():
    def __init__(self, odds: pd.DataFrame, current_season: str, months_of_form_list: list, league: str, data_co_uk_ulr: str, data_co_uk_path: str, data_co_uk_dict: dict, fixtures_path: str, fixtures_url: str, fixtures_dict: dict):
        self.odds = odds 
        self.league = league
        self.current_season = current_season 
        self.months_of_form_list = months_of_form_list
        self.data_co_uk_url = data_co_uk_ulr
        self.data_co_uk_path = data_co_uk_path
        self.data_co_uk_dict = data_co_uk_dict
        self.fixtures_url = fixtures_url
        self.fixtures_dict = fixtures_dict
        self.fixtures_path = fixtures_path
    
    def update_dataset(self, category: str):
        if category == 'odds':
            data_path, replacing_dict, url = self.data_co_uk_path, self.data_co_uk_dict, self.data_co_uk_url
            path = os.path.join(data_path,  f'E0-{self.current_season}.csv')
        
        elif category == 'fixtures':
            data_path, replacing_dict, url = self.fixtures_path, self.fixtures_dict, self.fixtures_url
            path = os.path.join(data_path,  f'SeasonFixtures.csv')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, "wb") as file:
                file.write(response.content)
            self.replace_team_names(data_path=path, replacing_dict=replacing_dict)
            logger.success(f"File downloaded and saved as {path}")
        else:
            logger.success(f"Failed to download the file. Status code: {response.status_code}")
            
    def replace_team_names(self, data_path, replacing_dict):
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
        
    def setup_upcoming_fixtures(self):
        fixtures = pd.read_csv(os.path.join(self.fixtures_path, 'SeasonFixtures.csv'))
        fixtures.rename(columns={'Home Team': 'HomeTeam', 'Away Team': 'AwayTeam'}, inplace=True)
        fixtures['Date'] = pd.to_datetime(fixtures['Date'], format="%d/%m/%Y %H:%M")
        fixtures['Date'] = fixtures['Date'].dt.strftime("%d/%m/%Y")
        
        logger.debug(self.odds)
        upcoming_matches = pd.merge(fixtures[['Date', 'HomeTeam', 'AwayTeam']], self.odds, left_on=['HomeTeam', 'AwayTeam'], right_on=['Home Team', 'Away Team'], how='inner')
        logger.debug(upcoming_matches)
        upcoming_matches = upcoming_matches.drop(columns=['Home Team', 'Away Team'], axis=1)
        upcoming_matches.rename(columns={'1': 'HomeWinOdds', 'x': 'DrawOdds', '2': 'AwayWinOdds', 'OverLine': 'OverLineOdds', 'UnderLine': 'UnderLineOdds'}, inplace=True)
        upcoming_matches.drop(columns=['Yes', 'No'])
        upcoming_fixtures_path = os.path.join(self.fixtures_path, 'UpcomingFixtures.csv')
        upcoming_matches.to_csv(upcoming_fixtures_path, index=False)
        logger.success(f'Upcoming fixtures saved succesfully at {upcoming_fixtures_path}')
        raw_short_term_path = os.path.join(self.fixtures_path, 'raw_files/ShortTermForm/')
        raw_long_term_path = os.path.join(self.fixtures_path, 'raw_files/LongTermForm/')
        preprocessed_path = os.path.join(self.fixtures_path, 'preprocessed_files/')
        for path in [raw_short_term_path, raw_long_term_path, preprocessed_path]:
            path_handler = PathHandler(path)
            path_handler.create_paths_if_not_exists()
        understat_parser = Understat_Parser(league=self.league, dictionary=self.data_co_uk_dict)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(understat_parser.get_upcoming_match_stats(current_season= self.current_season, months_of_form_list= self.months_of_form_list, upcoming_fixtures_path= self.fixtures_path))
            

        preprocessor = Preprocessor(preprocessed_data_path=preprocessed_path)
        upcoming_raw_short = preprocessor.get_files(folder_path=raw_short_term_path)
        upcoming_raw_long = preprocessor.get_files(folder_path=raw_long_term_path)
        upcoming_preprocessed_list = preprocessor.preprocessing_pipeline(data = [upcoming_raw_long, upcoming_raw_short])
        preprocessor.output_files(upcoming_preprocessed_list)
        
        
