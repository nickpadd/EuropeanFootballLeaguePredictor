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
    """A model that combines regressor models for home and away side goal rate prediction.

    Args:
        BaseModel : A base model upon which FootballPredictor is built.
    """
    def __init__(self) -> None:
        """Initializing the home_side and away_side predictors as None.
        """
        self.home_side = None
        self.away_side = None
    
    def build_model(self, regressor) ->None:
        """Making home_side and away_side instances of the regressor. In case of PoissonRegressor sets max_iter = 1000 to ensure convergance.

        Args:
            regressor (class): The regressor class.
        """
        self.home_side = regressor()
        self.away_side = regressor()
        
        if isinstance(self.home_side, PoissonRegressor):
            self.home_side = regressor(max_iter = 1000)
            self.away_side = regressor(max_iter = 1000)
        

    
    def train_model(self, train_data, home_goals, away_goals) -> None:
        """Trains the regressors 

        Args:
            train_data (np.array): An array of the normalized train dataset X.
            home_goals (np.array): An array of the target values Y of the home_side predictor.
            away_goals (np.array): An array of the target values Y of the away_side predictor.
        """
        self.home_side.fit(train_data, home_goals.ravel())
        self.away_side.fit(train_data, away_goals.ravel())
    
    def evaluate(self):
        """Not utilized
        """
        pass 
    
    def predict(self, data_for_prediction) -> dict:
        """Predicts the input data_for_prediction.

        Args:
            data_for_prediction (np.array): A normalized array in the format of train_data that is given for prediction to the regressor models.

        Returns:
            dict: A dictionary with the prediction of the model for 'home', 'away' goal rate. Minimum of 0 for both.
        """
        return {'home': np.maximum(self.home_side.predict(data_for_prediction), 0), 'away': np.maximum(self.away_side.predict(data_for_prediction), 0)}


class ProbabilityEstimatorNetwork:
    """A network of two FootballPredictor objects that is responsible for predicting the probabilities of certain outcomes.
    """
    def __init__(self, voting_dict: dict, matchdays_to_drop: int) -> None:
        """Initializing the network

        Args:
            voting_dict (dict): A dictionary that provides the weights of long_term_from and short_term_form
            matchdays_to_drop (int): The matchdays at the start of the season that are considered to provide redundant information to the model because the league table is not yet indicative of the teams performance due to small sample size.
        """
        self.short_term_model = FootballPredictor()
        self.long_term_model = FootballPredictor()
        self.voting_dict = voting_dict
        self.matchdays_to_drop = matchdays_to_drop
    
    def build_network(self, regressor) -> None:
        """Building the network

        Args:
            regressor (class): The regressor class.
        """
        self.short_term_model.build_model(regressor)
        self.long_term_model.build_model(regressor)
        
    def drop_matchdays(self, long_term_data: pd.DataFrame, short_term_data: pd.DataFrame) -> tuple:
        """Dropping the number of matchdays at the start of each season.

        Args:
            long_term_data (pd.DataFrame): A dataframe with the long term form team statistics.
            short_term_data (pd.DataFrame): A dataframe with the short term form team statistics.

        Returns: 
            tuple:
                pd.DataFrame: A dataframe with the statistics of the long term form, having filtered out the number of matchdays to drop.
                pd.DataFrame: A dataframe with the statistics of the short term form, having filtered out the number of matchdays to drop.
        """
        filtered_long_term_data = long_term_data[(long_term_data['HM'] > self.matchdays_to_drop)&((long_term_data['AM'] > self.matchdays_to_drop))]
        filtered_short_term_data = short_term_data[short_term_data['Match_id'].isin(filtered_long_term_data['Match_id'])]
        return filtered_long_term_data, filtered_short_term_data

    def normalize_array(self, array) -> np.array:
        """Normalizes the input array to [0, 1]

        Args:
            array (np.array): The array to be normalized

        Returns:
            np.array: The normalized array with values in [0, 1]
        """
        scaler = MinMaxScaler()
        normalized_array = scaler.fit_transform(array)
        return normalized_array
    
    def prepare_for_prediction(self, short_term_data: pd.DataFrame, long_term_data: pd.DataFrame, for_prediction_short: pd.DataFrame, for_prediction_long: pd.DataFrame) -> tuple:
        """Gets the datasets in the form loaded from the .csv files and prepares them for prediction. Calls drop_matchdays() and normalize_array().

        Args:
            short_term_data (pd.DataFrame): A dataframe with the unpreprocessed short term form team statistic to train from.
            long_term_data (pd.DataFrame): A dataframe with the unpreprocessed long term form team statistics to train from.
            for_prediction_short (pd.DataFrame): A dataframe with the preprocessed short term form team statistic to predict.
            for_prediction_long (pd.DataFrame): A dataframe with the preprocessed long term form team statistic to predict.

        Returns:
            tuple:
                np.array: An array of long term form data prepared the model training
                np.array: An array of short term form data prepared the model training
                np.array: An array of home goals
                np.array: An array of away goals
                pd.DataFrame: A dataframe containing match information for the training data
                pd.DataFrame: A dataframe containing match information for the for-prediction data
                np.array: An array of short term form data prepared for model prediction
                np.array: An array of long term form data prepared for model prediction
                
            **ATTENTION** The training data and for prediction data have slightly different naming columns due to the for prediction data being scraped and having a flactuating under/over line.
        """
        long_term_data, short_term_data = self.drop_matchdays(long_term_data=long_term_data, short_term_data=short_term_data)
        match_info = long_term_data[['Match_id', 'Date', 'HomeTeam', 'AwayTeam', 'Result', 'HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverOdds', 'UnderOdds']]
        try:
            for_prediction_info = for_prediction_long[['Match_id', 'Date', 'HomeTeam', 'AwayTeam', 'HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'Line', 'OverLineOdds', 'UnderLineOdds', 'Yes', 'No']]
        except KeyError:
            for_prediction_info = for_prediction_long[['Match_id', 'Date', 'HomeTeam', 'AwayTeam']]
            
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
        
    def train_network(self, short_term_data: np.array, long_term_data: np.array, home_goals: np.array, away_goals: np.array) -> None:
        """Trains the network using the input data

        Args:
            short_term_data (np.array): An array containing the short term form training data
            long_term_data (np.array): An array containing the long term form training data
            home_goals (np.array): An array containing the home goals training target values
            away_goals (np.array): An array containing the away goals training target values
        """
        self.short_term_model.train_model(train_data = short_term_data, home_goals = home_goals, away_goals = away_goals)
        self.long_term_model.train_model(train_data = long_term_data, home_goals = home_goals, away_goals = away_goals)
    
    def deduct_goal_rate(self, for_prediction_short_form: np.array, for_prediction_long_form: np.array) -> dict:
        """Predicts the home and away goal rate using the short/long term prediction models for home and away side.

        Args:
            for_prediction_short_form (np.array): An array of short term form data prepared for model prediction
            for_prediction_long_form (np.array): An array of long term form data prepared for model prediction

        Returns:
            dict: A dictionary containing 'home' and 'away' side goal rate values accessible by their respective keys. The goal rate is deducted by a weighted average with the weights provided by the user in the configuration file.
        """
        short_term_prediction  = self.short_term_model.predict(for_prediction_short_form)
        long_term_prediction = self.long_term_model.predict(for_prediction_long_form)
        return {'home': (self.voting_dict['short_term']*short_term_prediction['home'] + self.voting_dict['long_term']*long_term_prediction['home']).flatten(), 'away': (self.voting_dict['short_term']*short_term_prediction['away'] + self.voting_dict['long_term']*long_term_prediction['away']).flatten()}
    
    def get_scoreline_probabilities(self, home_goal_rate_array: np.array, away_goal_rate_array: np.array) -> list:
        """Gets the unique scoreline probabilities by using the Poisson mass probability function for the deducted home and away goal rates.

        Args:
            home_goal_rate_array (np.array): An array containing the goal rates for the home sides of the predicted matches
            away_goal_rate_array (np.array): An array containing the goal rates for the away sides of the predicted matches

        Returns:
            list: A list of arrays. Each array contains each scoreline predicted probability such that scoreline_array[home_goals][away_goals] = Probability_of_scoreline(home_goals-away_goals)
        """
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
    
    def get_betting_probabilities(self, scoreline_prob_list: list) -> list:
        """Gets the probabilities for the most popular betting results using the scoreline probability arrays

        Args:
            scoreline_prob_list (list): A list containing an array of the scoreline probabilities for each predicted match.

        Returns:
            list: A list of dictionaries. Each dictionary corresponds to a predicted match and contains the predicted probability of the respective bet shown by the key.
        """
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
            over4 = 0.0
            under4 = 0.0
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
                    if (away_goals+home_goals>=5):
                        over4 += scoreline_prob_array[home_goals, away_goals]
                    if(away_goals+home_goals<5):
                        under4 += scoreline_prob_array[home_goals, away_goals]
                    if (away_goals==0) or (home_goals==0): 
                        ng += scoreline_prob_array[home_goals, away_goals]
                    if (away_goals!=0) and (home_goals!=0):
                        gg += scoreline_prob_array[home_goals, away_goals]
            
            betting_probabilities_list.append({'home': home_win, 'draw': draw, 'away': away_win, 'over2.5': over2, 'under2.5': under2, 'over3.5': over3, 'under3.5': under3, 'over4.5': over4, 'under4.5': under4, 'over1.5': over1, 'under1.5': under1, 'ng': ng, 'gg': gg})
            
        return betting_probabilities_list    
    
    def get_prediction_dataframe(self, for_prediction_info: pd.DataFrame, scoreline_probabilities: np.array, betting_probabilities: dict) -> pd.DataFrame:
        """Produces a dataframe with the predicted probabilities for each match

        Args:
            for_prediction_info (pd.DataFrame): A dataframe containing match information for the for-prediction data
            scoreline_probabilities (np.array): A list of arrays. Each array contains each scoreline predicted probability such that scoreline_array[home_goals][away_goals] = Probability_of_scoreline(home_goals-away_goals)
            betting_probabilities (dict): A list of dictionaries. Each dictionary corresponds to a predicted match and contains the predicted probability of the respective bet shown by the key.

        Returns:
            pd.DataFrame: A dataframe with each row corresponding to a predicted match, containing the predicted probabilities
        """
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
            prediction_dataframe.loc[index, 'Over4.5Probability'] = bet['over4.5']
            prediction_dataframe.loc[index, 'Under4.5Probability'] = bet['under4.5']
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
    
    def produce_probabilities(self, short_term_data: pd.DataFrame, long_term_data: pd.DataFrame, for_prediction_short: pd.DataFrame, for_prediction_long: pd.DataFrame) -> pd.DataFrame:
        """A pipeline of functions that gets the data from the loaded files and outputs the prediction dataframe.

        Args:
            short_term_data (pd.DataFrame): A dataframe containing the short term form data to train the model, as read from the .csv file
            long_term_data (pd.DataFrame): A dataframe containing the training long term form data to train the model, as read from the .csv file
            for_prediction_short (pd.DataFrame): A dataframe containing the short term form data for prediction, as read from the .csv file
            for_prediction_long (pd.DataFrame): A dataframe containing the long term form data for prediction, as read from the .csv file

        Returns:
            pd.DataFrame: A dataframe with each row corresponding to a predicted match, containing the predicted probabilities
        """
        long_term_array, short_term_array, home_goals_array, away_goals_array, match_info, for_prediction_info, for_prediction_short_array, for_prediction_long_array = self.prepare_for_prediction(short_term_data=short_term_data, long_term_data=long_term_data, for_prediction_short=for_prediction_short, for_prediction_long=for_prediction_long)
        self.train_network(short_term_data=short_term_array, long_term_data=long_term_array, home_goals=home_goals_array, away_goals=away_goals_array)
        goal_rate = self.deduct_goal_rate(for_prediction_long_form=for_prediction_short_array, for_prediction_short_form=for_prediction_long_array)
        scoreline_probabilities = self.get_scoreline_probabilities(home_goal_rate_array = goal_rate['home'], away_goal_rate_array = goal_rate['away'])
        betting_probabilities = self.get_betting_probabilities(scoreline_prob_list=scoreline_probabilities)
        prediction_dataframe = self.get_prediction_dataframe(for_prediction_info= for_prediction_info, scoreline_probabilities= scoreline_probabilities, betting_probabilities= betting_probabilities)
        return prediction_dataframe
    
    def add_dummy_validation_columns(self, validation_data: dict) -> pd.DataFrame:
        """Adding dummy columns to the given dataframes in order to reproduce the format of the for-prediction scraped data the model is prepared to predict. Needed for the evaluation process as the evaluation uses historic odds from csv files and not scraped data.

        Args:
            validation_data (dict): A dataframe with match info and statistics in the format of the training dataframes

        Returns:
            pd.DataFrame: A dataframe with match info and statistics in the format of for-prediction dataframes, having changed the format and added dummy columns
        """
        for key in ['short_term', 'long_term']:
            validation_data[key].loc[:, 'Line'] = '2.5'
            validation_data[key].loc[:, 'Yes'] = None 
            validation_data[key].loc[:, 'No'] = None 
            validation_data[key] = validation_data['short_term'].rename(columns={'OverOdds':'OverLineOdds', 'UnderOdds':'UnderLineOdds'})

        return validation_data
    
    def remove_dummy_columns(self, prediction_dataframe: pd.DataFrame) -> pd.DataFrame:
        """An function that reverses the format of add_dummy_validation_columns()

        Args:
            prediction_dataframe (pd.DataFrame): A dataframe with match info and statistics in the format of for-prediction dataframes

        Returns:
            pd.DataFrame: A dataframe with match info and statistics in the format of the training dataframes
        """
        prediction_dataframe.drop(columns=['Line', 'Yes', 'No'], inplace=True)
        prediction_dataframe = prediction_dataframe.rename(columns={'OverLineOdds':'OverOdds', 'UnderLineOdds':'UnderOdds'})
        return prediction_dataframe
    
    def evaluate_per_season(self, validation_season: str, short_term_data: pd.DataFrame, long_term_data: pd.DataFrame, bettor: Bettor, evaluation_output: str) -> tuple:
        """An evaluation pipeline that gets the datasets loaded from .csv files and produces evaluation metrics and figures of the input validation_season

        Args:
            validation_season (str): The season to evaluate on. One of the available ['2017', '2018', '2019', '2020', '2021', '2022', '2023']
            short_term_data (pd.DataFrame): The dataset as loaded from the .csv file of short term form
            long_term_data (pd.DataFrame): The dataset as loaded from the .csv file of long term form
            bettor (Bettor): A bettor object
            evaluation_output (str): The path of the evaluation output figures to be saved at

        Returns:
            tuple:
                dict: A dictionary of figures. Each figure corresponds to the results of the bettor in certain betting categories specified by the dictionary keys
                dict: A dictionary of metrics in the format of {'Investment': initial investment value, 'NetGain': resulting net gain, 'ROI': resulting return of investment}
        """
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

    
    def cut_validation_season(self, short_term_data: pd.DataFrame, long_term_data: pd.DataFrame, validation_season: str):
        """Gets the short/long term form data as loaded from the .csv files and cuts the season to evaluate as specified by validation_season

        Args:
            short_term_data (pd.DataFrame): A dataframe containing the short term form data as loaded from the .csv file
            long_term_data (pd.DataFrame): A dataframe containing the long term form data as loaded from the .csv file
            validation_season (str): The specified season to cut out of the given datasets for evaluation. One of the available ['2017', '2018', '2019', '2020', '2021', '2022', '2023']

        Returns:
            tuple: 
                dict: A dictionary with keys corresponding to long/short term form that each contain the pd.DataFrame of the training data
                dict: A dictionary with keys corresponding to long/short term form that each contain the pd.DataFrame of the data for evaluation
        """
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