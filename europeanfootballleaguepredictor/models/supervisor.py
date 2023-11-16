from europeanfootballleaguepredictor.models.bettor import Bettor
from europeanfootballleaguepredictor.common.config_parser import Config_Parser
from sklearn.svm import SVC
import pandas as pd
from tqdm import tqdm
import numpy as np
from loguru import logger

class Supervisor:
    def __init__(self):
        self.classifier = {}
        self.approval = {}
        for bet in ['home_win', 'draw', 'away_win', 'over2.5', 'under2.5']:
            self.classifier[bet] = SVC()
            self.approval[bet] = {}

    
    def train(self, bets: pd.DataFrame, odds, results: dict):
        training_portions_list = []
        bookmaker_odds_list = []
        result_list = []
        for bet_name, odds_name in tqdm(zip(['home_win', 'draw', 'away_win', 'over2.5', 'under2.5'], ['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverOdds', 'UnderOdds']), total=5):
            for index, match_id in enumerate(bets['Match_id'].reset_index(drop=True)):
                training_portions_list.append(bets[bets['Match_id']==match_id][f'{bet_name}_portion'])
                bookmaker_odds_list.append(odds.loc[index, odds_name])
                result_list.append(int(results[match_id][bet_name]))
            
            self.classifier[bet_name].fit(np.column_stack((training_portions_list, bookmaker_odds_list)), result_list)
    
    def examine_bets(self, bets: pd.DataFrame, odds):
        for bet_name, odds_name in tqdm(zip(['home_win', 'draw', 'away_win', 'over2.5', 'under2.5'], ['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverOdds', 'UnderOdds']), total=5):
            for index, match_id in enumerate(bets['Match_id'].reset_index(drop=True)):
                portion_value = bets[bets['Match_id']==match_id][f'{bet_name}_portion']
                odds_value = odds.loc[index, odds_name]
                self.approval[bet_name][match_id] = self.classifier[bet_name].predict(np.column_stack((portion_value, odds_value)))
    