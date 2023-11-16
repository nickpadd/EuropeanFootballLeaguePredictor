from europeanfootballleaguepredictor.common.config_parser import Config_Parser
from europeanfootballleaguepredictor.utils.path_handler import PathHandler
from europeanfootballleaguepredictor.data.preprocessor import Preprocessor
import os 
from loguru import logger
import argparse
import sys


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

    path_handler = PathHandler(config.preprocessed_data_path)
    path_handler.create_paths_if_not_exists()
    preprocessor = Preprocessor(preprocessed_data_path = config.preprocessed_data_path)

    long_term_form_dataframe = preprocessor.get_files(os.path.join(config.raw_data_path, 'LongTermForm'))
    short_term_form_dataframe = preprocessor.get_files(os.path.join(config.raw_data_path, 'ShortTermForm'))

    dataframe_list = preprocessor.preprocessing_pipeline([long_term_form_dataframe, short_term_form_dataframe])
    preprocessor.output_files(dataframe_list)

    
    

if __name__ == "__main__":
    main()