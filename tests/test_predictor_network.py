from europeanfootballleaguepredictor.models.probability_estimator import ProbabilityEstimatorNetwork
from europeanfootballleaguepredictor.common.config_parser import Config_Parser
from loguru import logger 
import pandas as pd
import argparse
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Set a seed for reproducibility
seed_value = 42  # You can choose any integer as the seed
np.random.seed(seed_value)

@pytest.fixture 
def create_dummy_dataframe() ->pd.DataFrame:
    """Creates a dummy dataframe in the format of the gathered preprocessed datasets that are input to the probability_estimator_network

    Returns:
        pd.DataFrame: A dummy dataframe in the format of the gathered preprocessed datasets
    """
    dummy_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='D'),
        'HomeTeam': np.random.choice(['TeamA', 'TeamB', 'TeamC'], size=12),
        'AwayTeam': np.random.choice(['TeamX', 'TeamY', 'TeamZ'], size=12),
        'Result': [f"{np.random.randint(0, 5)}-{np.random.randint(0, 5)}" for _ in range(12)],
        'HomeWinOdds': np.round(np.random.uniform(1, 5, size=12), 2),
        'DrawOdds': np.round(np.random.uniform(1, 5, size=12), 2),
        'AwayWinOdds': np.round(np.random.uniform(1, 6, size=12), 2),
        'OverOdds': np.round(np.random.uniform(1, 3, size=12), 2),
        'UnderOdds': np.round(np.random.uniform(1, 3, size=12), 2),
        'HM': np.random.randint(0, 15, size=12),
        'HW/M': np.round(np.random.uniform(0.1, 0.9, size=12), 2),
        'HD/M': np.round(np.random.uniform(0.1, 0.9, size=12), 2),
        'HL/M': np.round(np.random.uniform(0.1, 0.9, size=12), 2),
        'HG/M': np.round(np.random.uniform(0.1, 2.5, size=12), 2),
        'HGA/M': np.round(np.random.uniform(0.1, 2.5, size=12), 2),
        'HPTS/M': np.round(np.random.uniform(0.1, 2.5, size=12), 2),
        'HxG/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'HNPxG/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'HxGA/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'HNPxGA/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'HNPxGD/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'HPPDA': np.round(np.random.uniform(5, 20, size=12), 2),
        'HOPPDA': np.round(np.random.uniform(5, 20, size=12), 2),
        'HDC/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'HODC/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'HxPTS/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'AM': np.random.randint(0, 15, size=12),
        'AW/M': np.round(np.random.uniform(0.1, 0.9, size=12), 2),
        'AD/M': np.round(np.random.uniform(0.1, 0.9, size=12), 2),
        'AL/M': np.round(np.random.uniform(0.1, 0.9, size=12), 2),
        'AG/M': np.round(np.random.uniform(0.1, 2.5, size=12), 2),
        'AGA/M': np.round(np.random.uniform(0.1, 2.5, size=12), 2),
        'APTS/M': np.round(np.random.uniform(0.1, 2.5, size=12), 2),
        'AxG/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'ANPxG/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'AxGA/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'ANPxGA/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'ANPxGD/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'APPDA': np.round(np.random.uniform(5, 20, size=12), 2),
        'AOPPDA': np.round(np.random.uniform(5, 20, size=12), 2),
        'ADC/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'AODC/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'AxPTS/M': np.round(np.random.uniform(0.5, 2.5, size=12), 2),
        'Match_id': [f"match_{i}" for i in range(1, 13)]
    })
    return dummy_data


class TestProbabilityEstimatorNetwork:
    """A class of tests of the probability_estimator_network
    """
    voting_dict = { 'long_term': 0.6, 'short_term': 0.4}
    matchdays_to_drop = 4
    tolerance = 0.03
    def test_build_network(self) ->None:
        """Tests weather the network is build correctly with the regression models having been initialized as expected class instances
        """
        lin = ProbabilityEstimatorNetwork(voting_dict=TestProbabilityEstimatorNetwork.voting_dict, matchdays_to_drop=TestProbabilityEstimatorNetwork.matchdays_to_drop)
        lin.build_network(LinearRegression)
        svr = ProbabilityEstimatorNetwork(voting_dict=TestProbabilityEstimatorNetwork.voting_dict, matchdays_to_drop=TestProbabilityEstimatorNetwork.matchdays_to_drop)
        svr.build_network(SVR)
        assert isinstance(lin.short_term_model.home_side, LinearRegression) and isinstance(lin.short_term_model.away_side, LinearRegression) and isinstance(lin.long_term_model.home_side, LinearRegression) and isinstance(lin.long_term_model.away_side, LinearRegression)
        assert isinstance(svr.short_term_model.home_side, SVR) and isinstance(svr.short_term_model.away_side, SVR) and isinstance(svr.long_term_model.home_side, SVR) and isinstance(svr.long_term_model.away_side, SVR)

    def test_drop_matchdays(self, create_dummy_dataframe :pd.DataFrame) -> None:
        """Tests the drop_matchdays method. Makes sure the length of the long/short form data is equal and that there are no matchdays that were supposed to be dropped in the resulting dataset

        Args:
            create_dummy_dataframe (pd.DataFrame): _description_
        """
        # Create sample data for testing
        long_term_data = create_dummy_dataframe
        short_term_data = create_dummy_dataframe
        
        network = ProbabilityEstimatorNetwork(voting_dict=TestProbabilityEstimatorNetwork.voting_dict, matchdays_to_drop=TestProbabilityEstimatorNetwork.matchdays_to_drop)
        filtered_long_term_data, filtered_short_term_data = network.drop_matchdays(long_term_data=long_term_data, short_term_data=short_term_data)
        assert all(filtered_long_term_data['AM'] > TestProbabilityEstimatorNetwork.matchdays_to_drop) and all(filtered_long_term_data['HM'] > TestProbabilityEstimatorNetwork.matchdays_to_drop)
        assert len(filtered_long_term_data) == len(filtered_short_term_data)
    
    def test_normalize_array(self) ->None:
        """Tests the normalize_array method. Makes sure the values of the array are all in [0, 1] as expected
        """
        network = ProbabilityEstimatorNetwork(voting_dict=TestProbabilityEstimatorNetwork.voting_dict, matchdays_to_drop=TestProbabilityEstimatorNetwork.matchdays_to_drop)
        # Test case 1: Positive values
        input_array = np.random.uniform(0, 10, size=(4, 4))
        normalized_array = network.normalize_array(input_array)
        assert ((normalized_array >= 0).all and (normalized_array <= 1).all)
        
    def test_get_scoreline_probabilities(self) ->None:
        """Tests get_scoreline_probabilities method. Makes sure that the sum of the distinct scoreline probabilities is close to 1 with a tolerance of atol
        """
        network = ProbabilityEstimatorNetwork(voting_dict=TestProbabilityEstimatorNetwork.voting_dict, matchdays_to_drop=TestProbabilityEstimatorNetwork.matchdays_to_drop)
        home_goal_rate = np.random.uniform(0, 5, size=10)
        away_goal_rate = np.random.uniform(0, 4, size=10)
        poisson_array_list = network.get_scoreline_probabilities(home_goal_rate_array= home_goal_rate, away_goal_rate_array= away_goal_rate)
        for array in poisson_array_list:
            assert np.isclose(np.round(np.sum(array), 2), 1, atol=self.tolerance)

    def test_get_betting_probabilities(self) ->None:
        """Tests get_betting_probabilities method. Makes sure that the sum of the distinct betting categories is close to 1 with a tolerance of atol
        """
        network = ProbabilityEstimatorNetwork(voting_dict=TestProbabilityEstimatorNetwork.voting_dict, matchdays_to_drop=TestProbabilityEstimatorNetwork.matchdays_to_drop)
        home_goal_rate = np.random.uniform(0, 5, size=10)
        away_goal_rate = np.random.uniform(0, 4, size=10)
        poisson_array_list = network.get_scoreline_probabilities(home_goal_rate_array= home_goal_rate, away_goal_rate_array= away_goal_rate)
        betting_probabilities_list = network.get_betting_probabilities(scoreline_prob_list=poisson_array_list)
        for prob_list in betting_probabilities_list:
            assert np.isclose(np.round(prob_list['home'] + prob_list['draw'] + prob_list['away'], 2), 1, atol=0.2)
            assert np.isclose(np.round(prob_list['over2.5'] + prob_list['under2.5'], 2), 1, atol=0.2)
            assert np.isclose(np.round(prob_list['over3.5'] + prob_list['under3.5'], 2), 1, atol=0.2)
            assert np.isclose(np.round(prob_list['gg'] + prob_list['ng'], 2), 1, atol=self.tolerance)
    
    def test_prepare_for_prediction(self, create_dummy_dataframe) ->None:
        """Tests the prepare_for_prediction method. Makessure the length of the sort/long form as well as home/away goals and match info is equal. Tests weather there are remaining null values in the resulting data

        Args:
            create_dummy_dataframe (func): The function that creates a dummy dataframe in the format of the collected preprocessed datasets
        """
        short, long, for_pred_short, for_pred_long = [create_dummy_dataframe for i in range(4)]
        
        for_pred_short['Yes']=np.round(np.random.uniform(1, 3, size=12), 2)
        for_pred_long['Yes']=np.round(np.random.uniform(1, 3, size=12), 2)
        for_pred_short['No'] =np.round(np.random.uniform(1, 3, size=12), 2)
        for_pred_long['No'] =np.round(np.random.uniform(1, 3, size=12), 2)
        for_pred_short['Line'] = np.random.choice(['2.5', '3.5'], size=12)
        for_pred_long['Line'] = np.random.choice(['2.5', '3.5'], size=12)
        for_pred_short = for_pred_short.rename(columns={'OverOdds': 'OverLineOdds', 'UnderOdds': 'UnderLineOdds'})
        for_pred_long = for_pred_long.rename(columns={'OverOdds': 'OverLineOdds', 'UnderOdds': 'UnderLineOdds'})

        network = ProbabilityEstimatorNetwork(voting_dict=TestProbabilityEstimatorNetwork.voting_dict, matchdays_to_drop=TestProbabilityEstimatorNetwork.matchdays_to_drop)
        long_term_array, short_term_array, home_goals_array, away_goals_array, match_info, for_prediction_info, for_prediction_short_array, for_prediction_long_array = network.prepare_for_prediction(short_term_data=short, long_term_data=long, for_prediction_long=for_pred_long, for_prediction_short=for_pred_short)
        assert len(long_term_array) == len(short_term_array) == len(home_goals_array) == len(away_goals_array) == len(match_info)
        assert len(for_prediction_info) == len(for_prediction_short_array) == len(for_prediction_long_array)
        assert not np.any(np.isnan(long_term_array))
        assert not np.any(np.isnan(short_term_array))
        assert not np.any(np.isnan(home_goals_array))
        assert not np.any(np.isnan(away_goals_array))
        assert not np.any(np.isnan(for_prediction_short_array))
        assert not np.any(np.isnan(for_prediction_long_array))
        assert not for_prediction_info.isna().any().any()
        assert not match_info.isna().any().any()       
    
    def test_deduct_goalrate(self) ->None:
        """Tests the deduct goalrate method. Makes sure that there are no null goal rate values
        """
        network = ProbabilityEstimatorNetwork(voting_dict=TestProbabilityEstimatorNetwork.voting_dict, matchdays_to_drop=TestProbabilityEstimatorNetwork.matchdays_to_drop)
        network.build_network(regressor = LinearRegression)
        
        train_short = np.round(np.random.uniform(0, 1, size=(12, 34)), 2)
        train_long = np.round(np.random.uniform(0, 1, size=(12, 34)), 2)
        for_pred_short = np.round(np.random.uniform(0, 1, size=(12, 34)), 2)
        for_pred_long = np.round(np.random.uniform(0, 1, size=(12, 34)), 2)
        home_goals = np.random.randint(0, 5, size=12)
        away_goals = np.random.randint(0, 5, size=12)
        
        
        network.train_network(short_term_data=train_short, long_term_data=train_long, home_goals=home_goals, away_goals=away_goals)
        goal_rate = network.deduct_goal_rate(for_prediction_long_form=for_pred_short, for_prediction_short_form=for_pred_long)
        logger.debug(goal_rate)
        assert not any(np.any(np.isnan(value)) for value in goal_rate.values())

    def test_produce_probabilities(self, create_dummy_dataframe) ->None:
        """Tests producce_probabilities method. Makes sure the sum of the the distinc betting category probabilities is close to 1 with a tolerance of atol

        Args:
            create_dummy_dataframe (func): The function that creates a dummy dataframe in the format of the collected preprocessed datasets
        """
        short, long, for_pred_short, for_pred_long = [create_dummy_dataframe for i in range(4)]
        
        for_pred_short['Yes']=np.round(np.random.uniform(1, 3, size=12), 2)
        for_pred_long['Yes']=np.round(np.random.uniform(1, 3, size=12), 2)
        for_pred_short['No'] =np.round(np.random.uniform(1, 3, size=12), 2)
        for_pred_long['No'] =np.round(np.random.uniform(1, 3, size=12), 2)
        for_pred_short['Line'] = np.random.choice(['2.5', '3.5'], size=12)
        for_pred_long['Line'] = np.random.choice(['2.5', '3.5'], size=12)
        for_pred_short = for_pred_short.rename(columns={'OverOdds': 'OverLineOdds', 'UnderOdds': 'UnderLineOdds'})
        for_pred_long = for_pred_long.rename(columns={'OverOdds': 'OverLineOdds', 'UnderOdds': 'UnderLineOdds'})
        
        network = ProbabilityEstimatorNetwork(voting_dict=TestProbabilityEstimatorNetwork.voting_dict, matchdays_to_drop=TestProbabilityEstimatorNetwork.matchdays_to_drop)
        network.build_network(regressor = LinearRegression)
        prediction_frame = network.produce_probabilities(long_term_data=long, short_term_data=short, for_prediction_long=for_pred_long, for_prediction_short=for_pred_short)
        
        logger.debug(prediction_frame)
        for index, row in prediction_frame.iterrows():
            win_sum = row['HomeWinProbability'] + row['DrawProbability'] + row['AwayWinProbability']
            line2_sum = row['Under2.5Probability'] + row['Over2.5Probability']
            line3_sum = row['Under3.5Probability'] + row['Over3.5Probability']
            gg_sum = row['GGProbability'] + row['NGProbability']
            assert np.isclose(np.round(win_sum, 2), 1, atol=self.tolerance)
            assert np.isclose(np.round(line2_sum, 2), 1, atol=self.tolerance)
            assert np.isclose(np.round(line3_sum, 2), 1, atol=self.tolerance)
            assert np.isclose(np.round(gg_sum, 2), 1, atol=self.tolerance)
