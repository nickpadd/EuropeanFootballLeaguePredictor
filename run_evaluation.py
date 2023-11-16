import argparse 
from europeanfootballleaguepredictor.common.config_parser import Config_Parser
from loguru import logger
from europeanfootballleaguepredictor.models.PredictorNetwork import ProbabilityEstimatorNetwork
from europeanfootballleaguepredictor.models.bettor import Bettor
from europeanfootballleaguepredictor.models.supervisor import Supervisor
import os 
import pandas as pd 

class BettingNetwork:
    def __init__(self, bettor: Bettor, supervisor: Supervisor, probability_estimator_network: ProbabilityEstimatorNetwork, probability_regressor):
        self.bettor = bettor 
        self.supervisor = supervisor
        self.probability_estimator_network = probability_estimator_network
        self.probability_regressor = probability_regressor
    
    def build_network(self):
        self.probability_estimator_network.build_network(regressor=self.probability_regressor)
    
    def train_network(self, training_data):
        validation_data = self.add_dummy_validation_columns(training_data.copy())
        
        results = validation_data['long_term'][['Match_id', 'Result']]
        probability_dataframe = self.probability_estimator_network.produce_probabilities(short_term_data=training_data['short_term'], long_term_data=training_data['long_term'], for_prediction_short=validation_data['short_term'], for_prediction_long=validation_data['long_term'])
        probability_dataframe = self.remove_dummy_columns(prediction_dataframe=probability_dataframe)
        self.bettor.preprocess(prediction_dataframe=probability_dataframe, results= results)
        self.bettor.place_value_bets()
        self.supervisor.train(bets = self.bettor.value_bets, results = self.bettor.result_dict, odds=self.bettor.bookmaker_probabilities)
        self.bettor.reset_bank()
    
    def predict(self, prediction_data, training_data):
        prediction_data = self.add_dummy_validation_columns(prediction_data.copy())
        results = prediction_data['long_term'][['Match_id', 'Result']]
        probability_dataframe = self.probability_estimator_network.produce_probabilities(short_term_data=training_data['short_term'], long_term_data=training_data['long_term'], for_prediction_short=prediction_data['short_term'], for_prediction_long=prediction_data['long_term'])
        probability_dataframe = self.remove_dummy_columns(prediction_dataframe=probability_dataframe)
        self.bettor.preprocess(prediction_dataframe=probability_dataframe, results= results)
        unsupervised_metrics = self.bettor.place_value_bets()
        logger.info(f'Unsupervised metrics: {unsupervised_metrics}')
        self.supervisor.examine_bets(bets = self.bettor.value_bets, odds=self.bettor.bookmaker_probabilities)
        self.bettor.reset_bank()
        supervised_metrics = self.bettor.place_supervised_bets(acceptance_dict= self.supervisor.approval)
        logger.info(f'Supervised metrics: {supervised_metrics}')
        
    def add_dummy_validation_columns(self, validation_data: dict):
        for key in ['short_term', 'long_term']:
            validation_data[key].loc[:, 'Line'] = '2.5'
            validation_data[key].loc[:, 'Yes'] = None 
            validation_data[key].loc[:, 'No'] = None 
            validation_data[key] = validation_data[key].rename(columns={'OverOdds':'OverLineOdds', 'UnderOdds':'UnderLineOdds'})
        

        return validation_data
    
    def remove_dummy_columns(self, prediction_dataframe):
        prediction_dataframe.drop(columns=['Line', 'Yes', 'No'], inplace=True)
        prediction_dataframe = prediction_dataframe.rename(columns={'OverLineOdds':'OverOdds', 'UnderLineOdds':'UnderOdds'})
        return prediction_dataframe
    
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
    short_term_data = pd.read_csv(os.path.join(config.preprocessed_data_path, 'ShortTermForm.csv'))
    long_term_data = pd.read_csv(os.path.join(config.preprocessed_data_path, 'LongTermForm.csv'))
    
    probability_estimator_network = ProbabilityEstimatorNetwork(voting_dict=config.voting_dict, matchdays_to_drop=config.matchdays_to_drop)
    training_data, validation_data = probability_estimator_network.cut_validation_season(short_term_data=short_term_data, long_term_data =long_term_data, validation_season=validation_season)
    bettor = Bettor(bank=config.bettor_bank, margin_dictionary=config.bettor_margin_dict)
    supervisor = Supervisor()
    betting_network = BettingNetwork(bettor = bettor, supervisor = supervisor, probability_estimator_network=probability_estimator_network, probability_regressor=config.regressor)
    betting_network.build_network()
    betting_network.train_network(training_data=training_data)
    betting_network.predict(training_data=training_data, prediction_data=validation_data)

if __name__ == "__main__":
    main()