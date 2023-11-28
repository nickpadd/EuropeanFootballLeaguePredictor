from europeanfootballleaguepredictor.models.probability_estimator import ProbabilityEstimatorNetwork
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
from europeanfootballleaguepredictor.data.database_handler import DatabaseHandler 

"""
European Football League Predictor Script

This script uses a probability estimator network to predict outcomes in the specified European football league and visualizes the predictions.
"""

def main():
    """Main entry point for the script.

    This function orchestrates the entire process of predicting football outcomes and visualizing the results.
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

    pd.set_option('display.precision', 2)
    
    database_handler = DatabaseHandler(league=config.league, database=config.database)
    probability_estimator_network = ProbabilityEstimatorNetwork(voting_dict=config.voting_dict, matchdays_to_drop=config.matchdays_to_drop)
    probability_estimator_network.build_network(regressor = config.regressor)
    
    short_term_form, long_term_form, for_prediction_short, for_prediction_long = database_handler.get_data(table_names=["Preprocessed_ShortTermForm", "Preprocessed_LongTermForm", "Preprocessed_UpcomingShortTerm", "Preprocessed_UpcomingLongTerm"])
    
    probability_dataframe = probability_estimator_network.produce_probabilities(short_term_data=short_term_form, long_term_data=long_term_form, for_prediction_short=for_prediction_short, for_prediction_long=for_prediction_long)
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