from europeanfootballleaguepredictor.models.probability_estimator import FootballPredictor, ProbabilityEstimatorNetwork
import numpy as np
import pandas as pd
from loguru import logger 
from europeanfootballleaguepredictor.common.config_parser import Config_Parser
from europeanfootballleaguepredictor.utils.path_handler import PathHandler
from europeanfootballleaguepredictor.visualization.visualize import Visualizer
from pretty_html_table import build_table
import argparse
import os
from europeanfootballleaguepredictor.models.bettor import Bettor
from europeanfootballleaguepredictor.models.supervisor import Supervisor
import mlflow.sklearn
import tempfile

def main():
    '''Parsing the configuration file'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file (e.g., config.yaml)", default='europeanfootballleaguepredictor/config/config.yaml')
    parser.add_argument("--season", type=str, help="The season on which the validation betting will take place.", required=True)
    config_file_path = parser.parse_args().config
    validation_season = parser.parse_args().season
    
    config_data_parser = Config_Parser(config_file_path, None)
    config_data = config_data_parser.load_and_extract_yaml_section()
    config = config_data_parser.load_configuration_class(config_data)
    
    logger.info(config)
    '''End of the configuration file parsing'''
    with mlflow.start_run(run_name = f"{config.league} | {validation_season}") as run:
        
        net = ProbabilityEstimatorNetwork(voting_dict=config.voting_dict, matchdays_to_drop=config.matchdays_to_drop)
        net.build_network(regressor = config.regressor)
        # Log model parameters
        mlflow.log_param("Form Votes", config.voting_dict)
        mlflow.log_param('Margin Dict', config.bettor_margin_dict)
        mlflow.log_param('Regressor', config.regressor)

        short_term_form = pd.read_csv(os.path.join(config.preprocessed_data_path, 'ShortTermForm.csv'))
        long_term_form = pd.read_csv(os.path.join(config.preprocessed_data_path, 'LongTermForm.csv'))

        bettor = Bettor(bank=config.bettor_bank, margin_dictionary=config.bettor_margin_dict)
        figures, metrics = net.evaluate_per_season(short_term_data=short_term_form, long_term_data=long_term_form, validation_season=validation_season, bettor= bettor, evaluation_output = config.evaluation_output)
        logger.info(metrics)
        
        mlflow.log_metric(f"Investment", metrics['Investment'])
        # Log metrics
        for name in ['home_win', 'draw', 'away_win', 'over2.5', 'under2.5']:
            mlflow.log_metric(f"NetGain_{name}", metrics['NetGain'][f'{name}_gain'])
            mlflow.log_metric(f"ROI_{name}", metrics['ROI'][f'{name}_roi'])
       
        # Save log the artifact
        tmp = tempfile.NamedTemporaryFile(prefix='residuals-', suffix='.png')
        tmp_name = tmp.name

        for name, fig in figures.items():
            try:
              fig.savefig(tmp_name)
              mlflow.log_artifact(tmp_name, f"{name}.png")
            finally:
              tmp.close()
    
if __name__ == "__main__":
    main()