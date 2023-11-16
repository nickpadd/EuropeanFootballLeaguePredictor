from europeanfootballleaguepredictor.data.understat_gatherer import Understat_Parser
import asyncio 
import os
from europeanfootballleaguepredictor.utils.path_handler import PathHandler
from loguru import logger
from europeanfootballleaguepredictor.common.config_parser import Config_Parser
import argparse
from europeanfootballleaguepredictor.data.preprocessor import Preprocessor

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
    
    understat_parser = Understat_Parser(league = config.league, dictionary = config.data_co_uk_dictionary)
    
    for dir, months_of_form in zip(['LongTermForm', 'ShortTermForm'], config.months_of_form_list):
        logger.info(f'Gathering {months_of_form} month form data for seasons in {config.seasons_to_gather}')
        extended_dir = os.path.join(config.raw_data_path, dir)
        path_handler = PathHandler(extended_dir)
        path_handler.create_paths_if_not_exists()
        for season in config.seasons_to_gather:
            full_path = os.path.join(extended_dir, f'Raw_{months_of_form}_{season}.csv')
            loop = asyncio.get_event_loop()
            loop.run_until_complete(understat_parser.get_understat_season_to_csv(season = season, months_of_form = months_of_form, output_path = full_path, data_co_uk_path = config.data_co_uk_path))
    
        logger.success(f'Succesfully finished {months_of_form} month(s) form gathering.')
    
    logger.success('Succesfully gathered and saved the datasets.')
    
    path_handler = PathHandler(config.preprocessed_data_path)
    path_handler.create_paths_if_not_exists()
    preprocessor = Preprocessor(preprocessed_data_path = config.preprocessed_data_path)

    long_term_form_dataframe = preprocessor.get_files(os.path.join(config.raw_data_path, 'LongTermForm'))
    short_term_form_dataframe = preprocessor.get_files(os.path.join(config.raw_data_path, 'ShortTermForm'))

    dataframe_list = preprocessor.preprocessing_pipeline([long_term_form_dataframe, short_term_form_dataframe])
    preprocessor.output_files(dataframe_list)
    
if __name__ == "__main__":
    main()