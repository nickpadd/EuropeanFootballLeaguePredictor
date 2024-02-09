import pytest
from europeanfootballleaguepredictor.models.bettor import Bettor 
import pandas as pd 
import numpy as np 

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
        'HomeWinOdds': np.round(np.random.uniform(1, 5, size=12), 2),
        'DrawOdds': np.round(np.random.uniform(1, 5, size=12), 2),
        'AwayWinOdds': np.round(np.random.uniform(1, 6, size=12), 2),
        'OverOdds': np.round(np.random.uniform(1, 3, size=12), 2),
        'UnderOdds': np.round(np.random.uniform(1, 3, size=12), 2),
        'HomeWinProbability': np.round(np.random.uniform(0.05, 0.95, size=12), 2), 
        'DrawProbability': np.round(np.random.uniform(0.05, 0.95, size=12), 2), 
        'AwayWinProbability': np.round(np.random.uniform(0.05, 0.95, size=12), 2), 
        'Over2.5Probability': np.round(np.random.uniform(0.05, 0.95, size=12), 2), 
        'Under2.5Probability': np.round(np.random.uniform(0.05, 0.95, size=12), 2),
        'Match_id': [f"match_{i}" for i in range(1, 13)]
    })
    
    dummy_results = pd.DataFrame({'results': [f"{np.random.randint(0, 5)}-{np.random.randint(0, 5)}" for _ in range(12)]})
    dummy_results['Match_id'] = dummy_data['Match_id']
    return dummy_data, dummy_results
    

class TestBettor:
    """A class that tests the Bettor module
    """
    
    def assert_bettor_is_default(self, bettor):
        for category_bank in bettor.current_bankroll.values():
            assert category_bank == 100
        for roi_metric, netgain_metric in zip(bettor.ROI.values(), bettor.NetGain.values()):
            assert roi_metric == 0
            assert netgain_metric == 0
        assert bettor.starting_bank == 100
    
    def assert_bettor_isnot_default(self, bettor):
        for category_bank in bettor.current_bankroll.values():
            assert category_bank != 100
        for roi_metric, netgain_metric in zip(bettor.ROI.values(), bettor.NetGain.values()):
            assert roi_metric != 0
            assert netgain_metric != 0
        assert bettor.starting_bank == 100
        assert bettor.kelly_cap == 0.2
        
    @pytest.mark.parametrize("bettor", [Bettor(bank=100, kelly_cap=0.2)])
    def test_init(self, bettor) -> None:
        self.assert_bettor_is_default(bettor)
        
    @pytest.mark.parametrize("bettor", [Bettor(bank=100, kelly_cap=0.2)])    
    def test_reset_bank(self, bettor) -> None:
        for category_bank in bettor.current_bankroll.keys():
            bettor.current_bankroll[category_bank] = 200
        for roi_metric, netgain_metric in zip(bettor.ROI.keys(), bettor.NetGain.keys()):
            bettor.ROI[roi_metric] = 200
            bettor.NetGain[netgain_metric] = 200
            
        
        self.assert_bettor_isnot_default(bettor)
        bettor.reset_bank()
        self.assert_bettor_is_default(bettor)
        
    @pytest.mark.parametrize("bettor", [Bettor(bank=100, kelly_cap=0.2)])    
    def test_preprocess(self, bettor, create_dummy_dataframe):
        info_columns = ['Match_id', 'Date', 'HomeTeam', 'AwayTeam']
        bookmaker_columns =['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverOdds', 'UnderOdds']
        model_columns = ['HomeWinProbability', 'DrawProbability', 'AwayWinProbability', 'Over2.5Probability', 'Under2.5Probability']
        dummy_data, dummy_results = create_dummy_dataframe
        bettor.preprocess(prediction_dataframe=dummy_data, results=dummy_results)
        for new, columns_to_check in zip([bettor.info, bettor.bookmaker_probabilities, bettor.model_probabilities, bettor.results], [info_columns, bookmaker_columns+['Match_id'], model_columns+['Match_id'], ['results']]):
            assert isinstance(new, pd.DataFrame)
            assert not new.isna().any().any()
            assert all(col in new.columns for col in columns_to_check)
            
        self.assert_bettor_is_default
        
    @pytest.mark.parametrize("bettor", [Bettor(bank=100, kelly_cap=0.2)])    
    def test_quarter_kelly_criterion(self, bettor):
        odds = [1.01, 2, 3, 10, 1.5, 0, 1, -1]
        probabilities = [0.8, 0.52, 0.4, 0.5, 0.1, 0.3, 0.98, 0.4]
        quarter_kelly_result = [0, 0.01, 0.025, 0.1111, 0, 0, 0, 0]
        cap = bettor.kelly_cap
        names = ['home_win', 'draw', 'away_win', 'over2.5', 'under2.5']
        for name in names:
            for odd, probability, result in zip(odds, probabilities, quarter_kelly_result):
                bankroll_portion, bet = bettor.quarter_kelly_criterion(bookmaker_odds = odd, estimated_true_probability= probability, bet_name = name, kelly_cap = cap)
                capped_result = min(result, cap)
                print(f" capped_result, bankroll_portion: {capped_result, bankroll_portion}")
                assert 0 <= bankroll_portion <= cap
                assert 0 <= bet <= bettor.current_bankroll[f"{name}_bank"]*cap
                assert np.round(bankroll_portion, 4) == capped_result
                assert np.round(bet, 2) == np.round(bettor.current_bankroll[f"{name}_bank"]*capped_result, 2)
                