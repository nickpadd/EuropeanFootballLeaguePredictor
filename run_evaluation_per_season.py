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
    with mlflow.start_run(run_name = f"{_regressor_instance} | {config.league}") as run:
        
        net = ProbabilityEstimatorNetwork(voting_dict=config.voting_dict, matchdays_to_drop=config.matchdays_to_drop)
        net.build_network(regressor = config.regressor)
        # Log model parameters
        mlflow.log_param("Form Votes", config.voting_dict)
        mlflow.log_param('Margin Dict', config.bettor_margin_dict)
        mlflow.log_param('Regressor', _regressor_instance)

        short_term_form = pd.read_csv(os.path.join(config.preprocessed_data_path, 'ShortTermForm.csv'))
        long_term_form = pd.read_csv(os.path.join(config.preprocessed_data_path, 'LongTermForm.csv'))

        weighted_average_ROI = {'1x2': [], 'over/under 2.5': [], 'home_win': [], 'draw': [], 'away_win': [], 'over2.5': [], 'under2.5': []}
        weights = [1, 2, 3, 4, 5, 6]
        weights_sum = sum(weights)
        for validation_season, weight in zip(['2017', '2018', '2019', '2020', '2021', '2022'], [w/weights_sum for w in weights]):
          bettor = Bettor(bank=config.bettor_bank, margin_dictionary=config.bettor_margin_dict)
          figures, metrics = net.evaluate_per_season(short_term_data=short_term_form, long_term_data=long_term_form, validation_season=validation_season, bettor= bettor, evaluation_output = config.evaluation_output)
          logger.info(metrics)
          
          for bet_name in ['home_win', 'draw', 'away_win', 'over2.5', 'under2.5']:
            mlflow.log_metric(f"ROI_{validation_season}_{bet_name}", metrics['ROI'][f'{bet_name}_roi'])
          
          avg_1x2 = statistics.mean([metrics['ROI']['home_win_roi'], metrics['ROI']['draw_roi'], metrics['ROI']['away_win_roi']])
          avg_ou = statistics.mean([metrics['ROI']['over2.5_roi'], metrics['ROI']['under2.5_roi']])
          
          weighted_average_ROI['1x2'].append(weight*avg_1x2)
          weighted_average_ROI['over/under 2.5'].append(weight*avg_ou)
          weighted_average_ROI['home_win'].append(weight*metrics['ROI']['home_win_roi'])
          weighted_average_ROI['draw'].append(weight*metrics['ROI']['draw_roi'])
          weighted_average_ROI['away_win'].append(weight*metrics['ROI']['away_win_roi'])
          weighted_average_ROI['over2.5'].append(weight*metrics['ROI']['over2.5_roi'])
          weighted_average_ROI['under2.5'].append(weight*metrics['ROI']['under2.5_roi'])
          
          # Save log the artifact
          tmp = tempfile.NamedTemporaryFile(prefix='residuals-', suffix='.png')
          tmp_name = tmp.name
  
          for name, fig in figures.items():
              try:
                fig.savefig(tmp_name)
                mlflow.log_artifact(tmp_name, f"{validation_season}/{name}.png")
              finally:
                tmp.close()

        weighted_average_ROI['1x2'] = statistics.mean(weighted_average_ROI['1x2'])
        weighted_average_ROI['over/under 2.5'] = statistics.mean(weighted_average_ROI['over/under 2.5'])
        weighted_average_ROI['home_win'] = statistics.mean(weighted_average_ROI['home_win'])
        weighted_average_ROI['draw'] = statistics.mean(weighted_average_ROI['draw'])
        weighted_average_ROI['away_win'] = statistics.mean(weighted_average_ROI['away_win'])
        weighted_average_ROI['over2.5'] = statistics.mean(weighted_average_ROI['over2.5'])
        weighted_average_ROI['under2.5'] = statistics.mean(weighted_average_ROI['under2.5'])
        
        
    mlflow.log_metric("WeightedAverage_1x2_ROI", round(weighted_average_ROI['1x2'], 2))
    mlflow.log_metric("WeightedAverage_over/under 2.5_ROI", round(weighted_average_ROI['over/under 2.5'], 2))
    mlflow.log_metric("WeightedAverage_home_win_ROI", round(weighted_average_ROI['home_win'], 2))
    mlflow.log_metric("WeightedAverage_draw_ROI", round(weighted_average_ROI['draw'], 2))
    mlflow.log_metric("WeightedAverage_away_win_ROI", round(weighted_average_ROI['away_win'], 2))
    mlflow.log_metric("WeightedAverage_over2.5_ROI", round(weighted_average_ROI['over2.5'], 2))
    mlflow.log_metric("WeightedAverage_under2.5_ROI", round(weighted_average_ROI['under2.5'], 2))


        
    
if __name__ == "__main__":
    main()