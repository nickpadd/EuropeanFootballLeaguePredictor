from europeanfootballleaguepredictor.models.PredictorNetwork import ProbabilityEstimatorNetwork
from europeanfootballleaguepredictor.models.bettor import Bettor
import numpy as np
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.svm import SVR
import pandas as pd
from loguru import logger 
from europeanfootballleaguepredictor.common.config_parser import Config_Parser
from europeanfootballleaguepredictor.utils.path_handler import PathHandler
from europeanfootballleaguepredictor.visualization.visualize import Visualizer
from pretty_html_table import build_table
import argparse
import os


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
    pd.set_option('display.precision', 2)
    
    net = ProbabilityEstimatorNetwork(voting_dict=config.voting_dict, matchdays_to_drop=config.matchdays_to_drop)
    net.build_network(regressor = config.regressor)
    
    short_term_form = pd.read_csv(os.path.join(config.preprocessed_data_path, 'ShortTermForm.csv'))
    long_term_form = pd.read_csv(os.path.join(config.preprocessed_data_path, 'LongTermForm.csv'))
    for_prediction_short = pd.read_csv(os.path.join(config.fixture_download_path, 'preprocessed_files/ShortTermForm.csv'))
    for_prediction_long = pd.read_csv(os.path.join(config.fixture_download_path, 'preprocessed_files/LongTermForm.csv'))
    
    probability_dataframe = net.produce_probabilities(short_term_data=short_term_form, long_term_data=long_term_form, for_prediction_short=for_prediction_short, for_prediction_long=for_prediction_long)
    logger.info(f'\n {probability_dataframe}')

    visualizer = Visualizer(probability_dataframe)
    figure = visualizer.radar_scoreline_plot()
    
    # Save the interactive figure as an HTML file
    output_handler = PathHandler(path=f'Predictions/{config.league}')
    output_handler.create_paths_if_not_exists()
    html_table = build_table(probability_dataframe.drop(['ScorelineProbability'], axis=1), 'blue_light')
    with open(f'Predictions/{config.league}/PredictionTable.html', 'w') as f:
        f.write(html_table)
    figure.write_html(f'Predictions/{config.league}/InteractiveFigure.html')
    
if __name__ == "__main__":
    main()