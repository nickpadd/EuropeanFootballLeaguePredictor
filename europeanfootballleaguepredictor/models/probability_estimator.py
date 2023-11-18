from europeanfootballleaguepredictor.models.base_model import BaseModel
from europeanfootballleaguepredictor.models.bettor import Bettor
import numpy as np
from scipy.stats import poisson
import pandas as pd 
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
import json

class FootballPredictor(BaseModel):
    def __init__(self):
        self.home_sode = None
        self.away_side = None
    
    def build_model(self, regressor):
        self.home_side = regressor()
        self.away_side = regressor()
        
        if isinstance(self.home_side, PoissonRegressor):
            self.home_side = regressor(max_iter = 1000)
            self.away_side = regressor(max_iter = 1000)

    
    def train_model(self, train_data, home_goals, away_goals):
        self.home_side.fit(train_data, home_goals.ravel())
        self.away_side.fit(train_data, away_goals.ravel())
    
    def evaluate(self):
        pass 
    
    def predict(self, data_for_prediction):
        return {'home': np.maximum(self.home_side.predict(data_for_prediction), 0), 'away': np.maximum(self.away_side.predict(data_for_prediction), 0)}


class ProbabilityEstimatorNetwork:
    def __init__(self, voting_dict: dict, matchdays_to_drop: int):
        self.short_term_model = FootballPredictor()
        self.long_term_model = FootballPredictor()
        self.voting_dict = voting_dict
        self.matchdays_to_drop = matchdays_to_drop
    
    def build_network(self, regressor):
        self.short_term_model.build_model(regressor)
        self.long_term_model.build_model(regressor)
        
    def drop_matchdays(self, long_term_data: pd.DataFrame, short_term_data: pd.DataFrame):
        filtered_long_term_data = long_term_data[(long_term_data['HM'] > self.matchdays_to_drop)&((long_term_data['AM'] > self.matchdays_to_drop))]
        filtered_short_term_data = short_term_data[short_term_data['Match_id'].isin(filtered_long_term_data['Match_id'])]
        return filtered_long_term_data, filtered_short_term_data

    def normalize_array(self, array):
        scaler = MinMaxScaler()
        normalized_array = scaler.fit_transform(array)
        return normalized_array
    
    def prepare_for_prediction(self, short_term_data: pd.DataFrame, long_term_data: pd.DataFrame, for_prediction_short: pd.DataFrame, for_prediction_long: pd.DataFrame):
        long_term_data, short_term_data = self.drop_matchdays(long_term_data=long_term_data, short_term_data=short_term_data)
        match_info = long_term_data[['Match_id', 'Date', 'HomeTeam', 'AwayTeam', 'Result', 'HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverOdds', 'UnderOdds']]
        for_prediction_info = for_prediction_long[['Match_id', 'Date', 'HomeTeam', 'AwayTeam', 'HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'Line', 'OverLineOdds', 'UnderLineOdds', 'Yes', 'No']]
        match_info = match_info.copy()
        match_info.loc[:, 'HomeGoals'] = match_info['Result'].str.split('-').str[0].astype(int)
        match_info.loc[:, 'AwayGoals'] = match_info['Result'].str.split('-').str[1].astype(int)
        match_info = match_info.drop('Result', axis=1)
        
        home_goals_array = np.array(match_info[['HomeGoals']])
        away_goals_array = np.array(match_info[['AwayGoals']])
        
        #Keeping the necessary columns
        long_term_array = np.array(
                                    long_term_data[
                                       [
                                        'HM', 'HW/M', 'HD/M', 'HL/M', 'HG/M', 'HGA/M', 'HPTS/M', 'HxG/M', 'HNPxG/M', 'HxGA/M', 'HNPxGA/M', 'HNPxGD/M', 'HPPDA', 'HOPPDA', 'HDC/M', 'HODC/M', 'HxPTS/M',
                                       'AM', 'AW/M', 'AD/M', 'AL/M', 'AG/M', 'AGA/M', 'APTS/M', 'AxG/M', 'ANPxG/M', 'AxGA/M', 'ANPxGA/M', 'ANPxGD/M', 'APPDA', 'AOPPDA', 'ADC/M', 'AODC/M', 'AxPTS/M']
                                       ])

        short_term_array = np.array(
                                    short_term_data[
                                       [
                                        'HM', 'HW/M', 'HD/M', 'HL/M', 'HG/M', 'HGA/M', 'HPTS/M', 'HxG/M', 'HNPxG/M', 'HxGA/M', 'HNPxGA/M', 'HNPxGD/M', 'HPPDA', 'HOPPDA', 'HDC/M', 'HODC/M', 'HxPTS/M',
                                        'AM', 'AW/M', 'AD/M', 'AL/M', 'AG/M', 'AGA/M', 'APTS/M', 'AxG/M', 'ANPxG/M', 'AxGA/M', 'ANPxGA/M', 'ANPxGD/M', 'APPDA', 'AOPPDA', 'ADC/M', 'AODC/M', 'AxPTS/M']
                                       ])
        for_prediction_long_array = np.array(
                                    for_prediction_long[
                                       [
                                        'HM', 'HW/M', 'HD/M', 'HL/M', 'HG/M', 'HGA/M', 'HPTS/M', 'HxG/M', 'HNPxG/M', 'HxGA/M', 'HNPxGA/M', 'HNPxGD/M', 'HPPDA', 'HOPPDA', 'HDC/M', 'HODC/M', 'HxPTS/M',
                                        'AM', 'AW/M', 'AD/M', 'AL/M', 'AG/M', 'AGA/M', 'APTS/M', 'AxG/M', 'ANPxG/M', 'AxGA/M', 'ANPxGA/M', 'ANPxGD/M', 'APPDA', 'AOPPDA', 'ADC/M', 'AODC/M', 'AxPTS/M']
                                       ])                                       
        for_prediction_short_array = np.array(
                                    for_prediction_short[
                                       [
                                        'HM', 'HW/M', 'HD/M', 'HL/M', 'HG/M', 'HGA/M', 'HPTS/M', 'HxG/M', 'HNPxG/M', 'HxGA/M', 'HNPxGA/M', 'HNPxGD/M', 'HPPDA', 'HOPPDA', 'HDC/M', 'HODC/M', 'HxPTS/M',
                                        'AM', 'AW/M', 'AD/M', 'AL/M', 'AG/M', 'AGA/M', 'APTS/M', 'AxG/M', 'ANPxG/M', 'AxGA/M', 'ANPxGA/M', 'ANPxGD/M', 'APPDA', 'AOPPDA', 'ADC/M', 'AODC/M', 'AxPTS/M']
                                       ])          
        
        for array in [long_term_array, short_term_array, for_prediction_short_array, for_prediction_long_array]:
            array = self.normalize_array(array)

        return long_term_array, short_term_array, home_goals_array, away_goals_array, match_info, for_prediction_info, for_prediction_short_array, for_prediction_long_array
        
    def train_network(self, short_term_data, long_term_data, home_goals, away_goals):
        self.short_term_model.train_model(train_data = short_term_data, home_goals = home_goals, away_goals = away_goals)
        self.long_term_model.train_model(train_data = long_term_data, home_goals = home_goals, away_goals = away_goals)
    
    def deduct_goal_rate(self, for_prediction_short_form, for_prediction_long_form):
        short_term_prediction  = self.short_term_model.predict(for_prediction_short_form)
        long_term_prediction = self.long_term_model.predict(for_prediction_long_form)
        return {'home': (self.voting_dict['short_term']*short_term_prediction['home'] + self.voting_dict['long_term']*long_term_prediction['home']).flatten(), 'away': (self.voting_dict['short_term']*short_term_prediction['away'] + self.voting_dict['long_term']*long_term_prediction['away']).flatten()}
    
    def get_scoreline_probabilities(self, home_goal_rate_array: np.array, away_goal_rate_array: np.array):
        max_g = 12
        goal_values = np.arange(max_g + 1)
        
        poisson_home = np.zeros((len(goal_values), 1))
        poisson_away = np.zeros((1, len(goal_values)))
        poisson_array_list = []
        
        for home_rate, away_rate in zip(home_goal_rate_array, away_goal_rate_array):
            for goal in goal_values:
                poisson_home[goal, 0] = poisson.pmf(goal, home_rate).item()
                poisson_away[0, goal] = poisson.pmf(goal, away_rate).item()
            poisson_array = np.matmul(poisson_home, poisson_away)
            poisson_array_list.append(poisson_array)

        return poisson_array_list
    
    def get_betting_probabilities(self, scoreline_prob_list):
        betting_probabilities_list = []
        for scoreline_prob_array in scoreline_prob_list:
            rows = len(scoreline_prob_array)
            columns = len(scoreline_prob_array[0])
            draw = np.trace(scoreline_prob_array)
            away_win = 0.0
            home_win = 0.0
            over2 = 0.0
            under2 = 0.0
            over3 = 0.0
            under3 = 0.0
            over1 = 0.0
            under1 = 0.0
            gg = 0.0
            ng = 0.0 
            
            for away_goals in range(rows):
                for home_goals in range(columns):
                    if (home_goals>away_goals):
                        home_win += scoreline_prob_array[home_goals, away_goals]
                    if (away_goals>home_goals):
                        away_win += scoreline_prob_array[home_goals, away_goals]
                    if (away_goals+home_goals>=2):
                        over1 += scoreline_prob_array[home_goals, away_goals]
                    if(away_goals+home_goals<2):
                        under1 += scoreline_prob_array[home_goals, away_goals]
                    if (away_goals+home_goals>=3):
                        over2 += scoreline_prob_array[home_goals, away_goals]
                    if(away_goals+home_goals<3):
                        under2 += scoreline_prob_array[home_goals, away_goals]
                    if (away_goals+home_goals>=4):
                        over3 += scoreline_prob_array[home_goals, away_goals]
                    if(away_goals+home_goals<4):
                        under3 += scoreline_prob_array[home_goals, away_goals]
                    if (away_goals==0) or (home_goals==0): 
                        ng += scoreline_prob_array[home_goals, away_goals]
                    if (away_goals!=0) and (home_goals!=0):
                        gg += scoreline_prob_array[home_goals, away_goals]
            
            betting_probabilities_list.append({'home': home_win, 'draw': draw, 'away': away_win, 'over2.5': over2, 'under2.5': under2, 'over3.5': over3, 'under3.5': under3, 'over1.5': over1, 'under1.5': under1, 'ng': ng, 'gg': gg})
            
        return betting_probabilities_list    
    
    def get_prediction_dataframe(self, for_prediction_info: pd.DataFrame, scoreline_probabilities: np.array, betting_probabilities: dict):

        prediction_dataframe = for_prediction_info.copy()
        prediction_dataframe['ScorelineProbability'] = [[] for _ in range(len(for_prediction_info))]
        
        for index, scoreline, bet in zip(range(len(for_prediction_info)), scoreline_probabilities, betting_probabilities):
            prediction_dataframe.loc[index, 'HomeWinProbability'] = bet['home']
            prediction_dataframe.loc[index, 'DrawProbability'] = bet['draw']
            prediction_dataframe.loc[index, 'AwayWinProbability'] = bet['away']
            prediction_dataframe.loc[index, 'Over2.5Probability'] = bet['over2.5']
            prediction_dataframe.loc[index, 'Under2.5Probability'] = bet['under2.5']
            prediction_dataframe.loc[index, 'Over3.5Probability'] = bet['over3.5']
            prediction_dataframe.loc[index, 'Under3.5Probability'] = bet['under3.5']
            prediction_dataframe.loc[index, 'Over1.5Probability'] = bet['over1.5']
            prediction_dataframe.loc[index, 'Under1.5Probability'] = bet['under1.5']
            prediction_dataframe.loc[index, 'GGProbability'] = bet['gg']
            prediction_dataframe.loc[index, 'NGProbability'] = bet['ng']
            # Create a dictionary
            score_dict = {}

            # Populate the dictionary with coordinates and probabilities
            for home_goals in range(12):
                for away_goals in range(12):
                    key = f'{home_goals} - {away_goals}'
                    value = scoreline[home_goals, away_goals]
                    score_dict[key] = value
            
            prediction_dataframe.loc[index, 'ScorelineProbability'] = [score_dict]
            prediction_dataframe['Date'] = pd.to_datetime(prediction_dataframe['Date'], format='%d/%m/%Y')
            prediction_dataframe.sort_values(by='Date', inplace=True)
            prediction_dataframe['Date'] = prediction_dataframe['Date'].dt.strftime('%d/%m/%Y')
        
        return prediction_dataframe
    
    def produce_probabilities(self, short_term_data: pd.DataFrame, long_term_data: pd.DataFrame, for_prediction_short: pd.DataFrame, for_prediction_long: pd.DataFrame):
        long_term_array, short_term_array, home_goals_array, away_goals_array, match_info, for_prediction_info, for_prediction_short_array, for_prediction_long_array = self.prepare_for_prediction(short_term_data=short_term_data, long_term_data=long_term_data, for_prediction_short=for_prediction_short, for_prediction_long=for_prediction_long)
        self.train_network(short_term_data=short_term_array, long_term_data=long_term_array, home_goals=home_goals_array, away_goals=away_goals_array)
        goal_rate = self.deduct_goal_rate(for_prediction_long_form=for_prediction_short_array, for_prediction_short_form=for_prediction_long_array)
        scoreline_probabilities = self.get_scoreline_probabilities(home_goal_rate_array = goal_rate['home'], away_goal_rate_array = goal_rate['away'])
        betting_probabilities = self.get_betting_probabilities(scoreline_prob_list=scoreline_probabilities)
        prediction_dataframe = self.get_prediction_dataframe(for_prediction_info= for_prediction_info, scoreline_probabilities= scoreline_probabilities, betting_probabilities= betting_probabilities)
        return prediction_dataframe
    
    def add_dummy_validation_columns(self, validation_data: dict):
        for key in ['short_term', 'long_term']:
            validation_data[key].loc[:, 'Line'] = '2.5'
            validation_data[key].loc[:, 'Yes'] = None 
            validation_data[key].loc[:, 'No'] = None 
            validation_data[key] = validation_data['short_term'].rename(columns={'OverOdds':'OverLineOdds', 'UnderOdds':'UnderLineOdds'})

        return validation_data
    
    def remove_dummy_columns(self, prediction_dataframe):
        prediction_dataframe.drop(columns=['Line', 'Yes', 'No'], inplace=True)
        prediction_dataframe = prediction_dataframe.rename(columns={'OverLineOdds':'OverOdds', 'UnderLineOdds':'UnderOdds'})
        return prediction_dataframe
    
    def evaluate_per_season(self, validation_season: str, short_term_data: pd.DataFrame, long_term_data: pd.DataFrame, bettor: Bettor, evaluation_output: str):
        training_data, validation_data = self.cut_validation_season(short_term_data=short_term_data, long_term_data=long_term_data, validation_season=validation_season)
        validation_data = self.add_dummy_validation_columns(validation_data= validation_data)
        validation_data['short_term'] = validation_data['short_term'].copy()
        validation_data['long_term'] = validation_data['long_term'].copy()
    
        results = validation_data['long_term'][['Match_id', 'Result']]

        prediction_dataframe = self.produce_probabilities(short_term_data= training_data['short_term'], long_term_data= training_data['long_term'], for_prediction_short=validation_data['short_term'], for_prediction_long=validation_data['long_term'])
        prediction_dataframe = self.remove_dummy_columns(prediction_dataframe=prediction_dataframe)
        bettor.preprocess(prediction_dataframe=prediction_dataframe, results= results)
        
        metrics = bettor.place_value_bets()
        figures = bettor.produce_report_figures(validation_season = validation_season, evaluation_output=evaluation_output)
        return figures, metrics

    
    def cut_validation_season(self, short_term_data, long_term_data, validation_season):
        logger.warning('Validating for Ligue_1 season 2022 seems to be not working currently!')
        data = {'short_term': short_term_data, 'long_term': long_term_data}
        data['short_term']['Date'] = pd.to_datetime(data['short_term']['Date'], format='%d/%m/%Y')
        data['long_term']['Date'] = pd.to_datetime(data['long_term']['Date'], format='%d/%m/%Y')

        # Define the start and end dates of the validation season
        #This dating is because of the covid seasons in serie_a taking until 08-05 in 2019
        validation_start_date = pd.to_datetime(f'{validation_season}-08-06')
        validation_end_date = pd.to_datetime(f'{int(validation_season) + 1}-08-05')

        # Filter the dataset to include only the data within the validation season
        validation_data = {'short_term': data['short_term'][(pd.to_datetime(data['short_term']['Date'], format='%d/%m/%Y') >= validation_start_date) & (pd.to_datetime(data['short_term']['Date'], format='%d/%m/%Y') <= validation_end_date)].reset_index(drop=True),
                           'long_term': data['long_term'][(pd.to_datetime(data['long_term']['Date'], format='%d/%m/%Y') >= validation_start_date) & (pd.to_datetime(data['long_term']['Date'], format='%d/%m/%Y') <= validation_end_date)].reset_index(drop=True)}

        # Separate the training data (data before or after the validation season) and validation data
        training_data = {'short_term': data['short_term'][(pd.to_datetime(data['short_term']['Date'], format='%d/%m/%Y') < validation_start_date) | (pd.to_datetime(data['short_term']['Date'], format='%d/%m/%Y') > validation_end_date)].reset_index(drop=True),
                         'long_term': data['long_term'][(pd.to_datetime(data['long_term']['Date'], format='%d/%m/%Y') < validation_start_date) | (pd.to_datetime(data['long_term']['Date'], format='%d/%m/%Y') > validation_end_date)].reset_index(drop=True)}

        return training_data, validation_data