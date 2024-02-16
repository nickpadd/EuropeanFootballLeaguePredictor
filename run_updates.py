from europeanfootballleaguepredictor.data.bookmaker_scraper_v2 import BookmakerScraper
from europeanfootballleaguepredictor.data.understat_gatherer import Understat_Parser
from europeanfootballleaguepredictor.common.config_parser import Config_Parser
from europeanfootballleaguepredictor.data.upcoming_matches import UpcomingMatchScheduler
from europeanfootballleaguepredictor.utils.path_handler import PathHandler
from loguru import logger
import asyncio
import os
from europeanfootballleaguepredictor.data.preprocessor import Preprocessor
import argparse
import pandas as pd
        
"""
European Football League Predictor

This script performs data scraping and processing for predicting outcomes in the configuration specified European Football League.
"""

def main():
    """Main entry point for the script.

    This function orchestrates the entire data scraping and processing pipeline.
    """
    
    #Parsing the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file (e.g., config.yaml)", default='europeanfootballleaguepredictor/config/config.yaml')
    config_file_path = parser.parse_args().config
    
    # Loading and extracting configuration data
    config_data_parser = Config_Parser(config_file_path, None)
    config_data = config_data_parser.load_and_extract_yaml_section()
    config = config_data_parser.load_configuration_class(config_data)
    
    logger.info(config)
    

    understat_parser = Understat_Parser(league = config.league, dictionary = config.data_co_uk_dictionary, database=config.database)
    bookmaker_scraper = BookmakerScraper(url = config.bookmaker_url, dictionary = config.bookmaker_dictionary)
    try:
        odds_dataframe = bookmaker_scraper.return_odds()
        logger.success('Successfully retrieved odds!')
        logger.info(f'\n {odds_dataframe}')
    except Exception as e:
        logger.warning(f'Error while fetching bookmaker odds: {e}')
        logger.warning('The bookmaker url may not be accessible from your IP address. Try using a Greek IP vpn! Will proceed without scraped bookmaker odds!')
        odds_dataframe = None
    

    upcoming_match_scheduler = UpcomingMatchScheduler(
        league = config.league,
        odds = odds_dataframe, 
        current_season = config.current_season,
        months_of_form_list= config.months_of_form_list,
        data_co_uk_ulr= config.data_co_uk_url, 
        data_co_uk_dict= config.data_co_uk_dictionary, 
        fixtures_url = config.fixture_download_url,
        fixtures_dict = config.fixture_download_dictionary,
        database = config.database
        )
    upcoming_match_scheduler.update_dataset('odds')
    upcoming_match_scheduler.update_dataset('fixtures')
    upcoming_match_scheduler.setup_upcoming_fixtures()
    
    #Updating the UpcomingShortTerm and UpcomingLongTerm tables
    for name, months_of_form in zip(['Raw_LongTermForm', 'Raw_ShortTermForm'], config.months_of_form_list):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(understat_parser.get_understat_season(season = config.current_season, months_of_form = months_of_form, output_table_name=name))
    
    #Preprocessing season raw statistics   
    preprocessor = Preprocessor(league=config.league, database=config.database)
    long_term_form_season_list = preprocessor.database_handler.get_data([f'Raw_LongTermForm_Season{season}_{str(int(season)+1)}' for season in config.seasons_to_gather])
    short_term_form_season_list = preprocessor.database_handler.get_data([f'Raw_ShortTermForm_Season{season}_{str(int(season)+1)}' for season in config.seasons_to_gather])
    long_term_form_dataframe = pd.concat([dataframe for dataframe in long_term_form_season_list])
    short_term_form_dataframe = pd.concat([dataframe for dataframe in short_term_form_season_list])
    
    preprocessed_dataframes = preprocessor.preprocessing_pipeline(data=[long_term_form_dataframe, short_term_form_dataframe])
    preprocessor.database_handler.save_dataframes(dataframes=preprocessed_dataframes, table_names=['Preprocessed_LongTermForm', 'Preprocessed_ShortTermForm'])
    
if __name__ == "__main__":
    main()