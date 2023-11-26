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
import mlflow.sklearn
import tempfile
import statistics

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
    _regressor_instance = config.regressor()

        
    net = ProbabilityEstimatorNetwork(voting_dict=config.voting_dict, matchdays_to_drop=config.matchdays_to_drop)
    net.build_network(regressor = config.regressor)

    short_term_form = pd.read_csv(os.path.join(config.preprocessed_data_path, 'ShortTermForm.csv'))
    long_term_form = pd.read_csv(os.path.join(config.preprocessed_data_path, 'LongTermForm.csv'))

    for validation_season in ['2017', '2018', '2019', '2020', '2021', '2022']:
      with mlflow.start_run(run_name = f"{_regressor_instance} | {config.league} | {validation_season}") as run:
        # Log model parameters
        mlflow.log_param("Form_Votes", config.voting_dict)
        mlflow.log_param('Margin_Dict', config.bettor_kelly_cap)
        mlflow.log_param('Regressor', _regressor_instance)
        mlflow.log_param('League', config.league)
        mlflow.log_param('Matchdays to drop', config.matchdays_to_drop)
        bettor = Bettor(bank=config.bettor_bank, kelly_cap=config.bettor_kelly_cap)
        figures, metrics = net.evaluate_per_season(short_term_data=short_term_form, long_term_data=long_term_form, validation_season=validation_season, bettor= bettor, evaluation_output = config.evaluation_output)
        logger.info(metrics)
          
        #logging metrics as a table
        roi_df = pd.DataFrame(list(metrics['ROI'].items()), columns=['Metric', 'Value'])
        mlflow.log_table(data=roi_df.round(2), artifact_file=f"ROI")
          
        # Log metrics individualy
        for bet_name in ['home_win', 'draw', 'away_win', 'over2.5', 'under2.5']:
          mlflow.log_metric(f"ROI_{bet_name}", np.round(metrics['ROI'][f'{bet_name}_roi'], 2))
          
        roi_1x2 = statistics.mean([metrics['ROI']['home_win_roi'], metrics['ROI']['draw_roi'], metrics['ROI']['away_win_roi']])
        roi_12 = statistics.mean([metrics['ROI']['home_win_roi'], metrics['ROI']['away_win_roi']])
        roi_ou = statistics.mean([metrics['ROI']['over2.5_roi'], metrics['ROI']['under2.5_roi']])
        
        mlflow.log_metric(f"ROI_1x2", np.round(roi_1x2, 2))
        mlflow.log_metric(f'ROI_12', np.round(roi_12, 2))
        mlflow.log_metric(f"ROI_ou", np.round(roi_ou, 2))
          
        # Save log the artifact
        tmp = tempfile.NamedTemporaryFile(prefix='residuals-', suffix='.png')
        tmp_name = tmp.name
  
        for name, fig in figures.items():
            try:
              fig.savefig(tmp_name)
              mlflow.log_artifact(tmp_name, f"{validation_season}/{name}.png")
            finally:
              tmp.close()        
    
if __name__ == "__main__":
    main()