from europeanfootballleaguepredictor.common.config_parser import Config_Parser
from europeanfootballleaguepredictor.utils.path_handler import PathHandler
from europeanfootballleaguepredictor.data.preprocessor import Preprocessor
import os 
from loguru import logger
import argparse
import sys
import pandas as pd 

def main():
    """Main entry point for the data preprocessing script."""
    
    #Parsing the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file (e.g., config.yaml)", default='europeanfootballleaguepredictor/config/config.yaml')
    config_file_path = parser.parse_args().config
    
    #Loading the configuration settings
    config_data_parser = Config_Parser(config_file_path, None)
    config_data = config_data_parser.load_and_extract_yaml_section()
    config = config_data_parser.load_configuration_class(config_data)
    
    logger.info(config)
    
    preprocessor = Preprocessor(league=config.league, database=config.database)

    #Gathering all the seasons into two concatenated dataframes one for long term and one for short term form
    long_term_form_season_list = preprocessor.database_handler.get_data([f'Raw_LongTermForm_Season{season}_{str(int(season)+1)}' for season in config.seasons_to_gather])
    short_term_form_season_list = preprocessor.database_handler.get_data([f'Raw_ShortTermForm_Season{season}_{str(int(season)+1)}' for season in config.seasons_to_gather])
    long_term_form_dataframe = pd.concat([dataframe for dataframe in long_term_form_season_list])
    short_term_form_dataframe = pd.concat([dataframe for dataframe in short_term_form_season_list])
    
    preprocessed_dataframes = preprocessor.preprocessing_pipeline(data=[long_term_form_dataframe, short_term_form_dataframe])
    preprocessor.database_handler.save_dataframes(dataframes=preprocessed_dataframes, table_names=['Preprocessed_LongTermForm', 'Preprocessed_ShortTermForm'])
    
    
    

if __name__ == "__main__":
    main()