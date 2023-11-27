from europeanfootballleaguepredictor.data.bookmaker_scraper import BookmakerScraper
from europeanfootballleaguepredictor.data.understat_gatherer import Understat_Parser
from europeanfootballleaguepredictor.common.config_parser import Config_Parser
from europeanfootballleaguepredictor.data.upcoming_matches import UpcomingMatchScheduler
from europeanfootballleaguepredictor.utils.path_handler import PathHandler
from loguru import logger
import asyncio
import os
import argparse
        
        
def main():
    '''Parsing the configuration file'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file (e.g., config.yaml)", default='europeanfootballleaguepredictor/config/config.yaml')
    config_file_path = parser.parse_args().config
    
    config_data_parser = Config_Parser(config_file_path, None)
    config_data = config_data_parser.load_and_extract_yaml_section()
    config = config_data_parser.load_configuration_class(config_data)
    
    logger.info(config)
    '''End of the configuration file parsing'''
    
    logger.info('Continuing with bookmaker scraping.')
    understat_parser = Understat_Parser(league = config.league, dictionary = config.data_co_uk_dictionary)
    bookmaker_scraper = BookmakerScraper(url = config.bookmaker_url, dictionary = config.bookmaker_dictionary)
    try:
        odds_dataframe = bookmaker_scraper.get_odds()
        logger.success('Successfully retrieved odds!')
        logger.info(f'\n {odds_dataframe}')
    except:
        logger.warning('The bookmaker url may not be accessible from your IP address. Try using a Greek IP vpn! Will proceed without scraped bookmaker odds!')
        odds_dataframe = None
    
    
    upcoming_match_scheduler = UpcomingMatchScheduler(
        league = config.league,
        odds = odds_dataframe, 
        current_season = config.current_season,
        months_of_form_list= config.months_of_form_list,
        data_co_uk_ulr= config.data_co_uk_url, 
        data_co_uk_dict= config.data_co_uk_dictionary, 
        data_co_uk_path = config.data_co_uk_path,
        fixtures_url = config.fixture_download_url,
        fixtures_dict = config.fixture_download_dictionary,
        fixtures_path = config.fixture_download_path
        )
    upcoming_match_scheduler.update_dataset('odds')
    upcoming_match_scheduler.update_dataset('fixtures')
    upcoming_match_scheduler.setup_upcoming_fixtures()
    
    for dir, months_of_form in zip(['LongTermForm', 'ShortTermForm'], config.months_of_form_list):
        extended_dir = os.path.join(config.raw_data_path, dir)
        full_path = os.path.join(extended_dir, f'Raw_{months_of_form}_{config.current_season}.csv')
        handler = PathHandler(path= extended_dir)
        handler.create_paths_if_not_exists()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(understat_parser.get_understat_season_to_csv(season = config.current_season, months_of_form = months_of_form, output_path = full_path, data_co_uk_path = config.data_co_uk_path))

if __name__ == "__main__":
    main()