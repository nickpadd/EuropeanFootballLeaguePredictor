import numpy as np 
from loguru import logger 
import matplotlib.pyplot as plt
import seaborn as sns 
import os
from tqdm import tqdm 
from sklearn.svm import SVC
import pandas as pd

class Bettor:
    """A Bettor class to find and place value bets as well as give report on the results in a given set of matches
    """
    def __init__(self, bank: float, kelly_cap: float) -> None:
        """Initializes the bettor object

        Args:
            bank (float): The initial investment the bettor has in its disposal at the start of the betting for each different betting category 
            kelly_cap (float): The max percentage of the current bankroll to bet
        """
        self.starting_bank = bank
        self.current_bankroll = {
            'home_win_bank': bank,
            'draw_bank': bank,
            'away_win_bank': bank,
            'over2.5_bank': bank,
            'under2.5_bank': bank
        }
        self.ROI = {
            'home_win_roi': 0,
            'draw_roi': 0,
            'away_win_roi': 0,
            'over2.5_roi': 0,
            'under2.5_roi': 0
        }
        self.NetGain = {
            'home_win_gain': 0,
            'draw_gain': 0,
            'away_win_gain': 0,
            'over2.5_gain': 0,
            'under2.5_gain': 0
        }
        self.kelly_cap = kelly_cap
        self.result_dict = {}
    
    def reset_bank(self) -> None:
        """Resests the starting bank by re-initializing the bettor
        """
        self.__init__(bank= self.starting_bank, kelly_cap=self.kelly_cap)
        
    def preprocess(self, prediction_dataframe: pd.DataFrame, results: pd.DataFrame) -> None:    
        """Prepares the bettor for prediction by parsing the prediction dataframe

        Args:
            prediction_dataframe (pd.DataFrame): A dataframe containing match information, predicted model probabilities and bookmaker odds that the bettor uses to find and place value bets
            results (pd.DataFrame): A dataframe containing the resulting scoreline of each match
        """
        info_columns = ['Match_id', 'Date', 'HomeTeam', 'AwayTeam']
        bookmaker_columns =['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverOdds', 'UnderOdds']
        model_columns = ['HomeWinProbability', 'DrawProbability', 'AwayWinProbability', 'Over2.5Probability', 'Under2.5Probability']
        self.info = prediction_dataframe[info_columns]
        self.bookmaker_probabilities = prediction_dataframe[bookmaker_columns+['Match_id']]
        self.model_probabilities = prediction_dataframe[model_columns+['Match_id']]
        self.results = results.reset_index(drop=True)
        
    def quarter_kelly_criterion(self, bookmaker_odds: float, estimated_true_probability: float, bet_name: str, kelly_cap: float) -> tuple:
        """Returns the portion of the current bankroll to bet as well as the bet, using the quarter kelly criterion and taking into account the kelly cap as the maximum portion to bet, specified by the user in the configuration file

        Args:
            bookmaker_odds (float): A dataframe containing the bookmaker odds of the for-prediction matches
            estimated_true_probability (float): A dataframe containing the models estimated probabilities of popular bets of the for-prediction matches
            bet_name (str): The name of the bet the kelly criterion should consider, one of ['home_win', 'draw', 'away_win', 'over2.5', 'under2.5']
            kelly_cap (float): A float indicating the maximum portion of the current bankroll the criterion should output, should be (0, 1]. For example for kelly_cap=0.1 the maximum output bet would be 10% of the current bankroll.

        Returns:
            tuple:
                float: The portion of the current bankroll to bet
                float: The value of the bet
        """ 
        #QUARTER KELLY
        portion_of_bet_gained_with_win = bookmaker_odds - 1
        if portion_of_bet_gained_with_win <= 0:
            capped_bankroll_portion, capped_bet = 0, 0
        else:
            bankroll_portion = max(estimated_true_probability - (1 - estimated_true_probability)/portion_of_bet_gained_with_win, 0) #Normal kelly bankroll portion
            bet = max(bankroll_portion*self.current_bankroll[f'{bet_name}_bank'], 0) #Normal kelly bet

            capped_bankroll_portion = min(np.divide(bankroll_portion, 4), kelly_cap) #Quarter kelly and cap 
            capped_bet = capped_bankroll_portion*self.current_bankroll[f'{bet_name}_bank'] #Quarter kelly and cap 
        return capped_bankroll_portion, capped_bet
    
    def get_betting_result(self, match_id: str) -> dict:
        """Searches and returns for the result dictionary of the specified match

        Args:
            match_id (str): The id of the match to return the result dictionary

        Returns:
            dict: The result dictionary with keys ['scoreline', 'home_win', 'draw', 'away_win', 'over2.5', 'under2.5'] with a string in the format of '{home_goals}-{away_goals}' for scoreline and with True or False for the betting outcomes
        """
        scoreline = self.results.loc[self.results['Match_id']==match_id, 'Result'].values[0]
        home_goals = int(scoreline.split('-')[0])
        away_goals = int(scoreline.split('-')[1])
    
        result_dict = {
            'scoreline': scoreline,
            'home_win': True if home_goals>away_goals else False,
            'draw': True if home_goals==away_goals else False,
            'away_win': True if home_goals<away_goals else False,
            'over2.5': True if home_goals+away_goals>=3 else False,
            'under2.5': True if home_goals+away_goals<3 else False
        }
        self.result_dict[match_id] = result_dict
        return result_dict
    
    def pay_bet(self, bet: float, bet_name: str) -> None:
        """Pays the specified bet amount in the bank of the bet category

        Args:
            bet (float): The value of the bet that is to be paid
            bet_name (str): An identification of the bet category
        """
        self.current_bankroll[f'{bet_name}_bank'] -= bet
    
    def get_payed_if_won(self, bet: float, bet_name: str, result: bool, bookmaker_odds: float) -> None:
        """Gets payed in the specific bet category bank and updates the ROI and NetGain of the bettor

        Args:
            bet (float): The bet value
            bet_name (str): An identification of the bet category
            result (bool): The result of the bet [True, False]
            bookmaker_odds (float): The odds provided by the bookmaker
        """
        if result == True:
            self.current_bankroll[f'{bet_name}_bank'] += bet*bookmaker_odds
        elif result == False:
            pass
        
        self.NetGain[f'{bet_name}_gain'] = self.current_bankroll[f'{bet_name}_bank'] - self.starting_bank
        self.ROI[f'{bet_name}_roi'] = 100*self.NetGain[f'{bet_name}_gain']/self.starting_bank
            
    def place_value_bets(self) -> dict:
        """Conducts the search and placement as well as the payment of the bets in each betting category and gets the results

        Returns:
            dict: A dictionary with the results of the betting process. Each of the key values contains another dictionary with the specified information divided in the betting categories. 
        """
        logger.info('Betting on the season...')
        bet_columns = ['home_win', 'draw', 'away_win', 'over2.5', 'under2.5']
        value_bets = self.info.copy()
        for index, id in tqdm(enumerate(value_bets['Match_id']), total=len(value_bets['Match_id'])):
            result_dict = self.get_betting_result(match_id = id)
            for bookmaker_odds_column, model_probability_column, bet_name in zip(self.bookmaker_probabilities.columns, self.model_probabilities.columns, bet_columns):
                bookmaker_odds = self.bookmaker_probabilities.loc[self.bookmaker_probabilities['Match_id']==id, bookmaker_odds_column].values[0]
                model_probability = self.model_probabilities.loc[self.model_probabilities['Match_id']==id, model_probability_column].values[0]
                portion, bet = self.quarter_kelly_criterion(bookmaker_odds=bookmaker_odds, estimated_true_probability=model_probability, bet_name=bet_name, kelly_cap=self.kelly_cap)
                value_bets.loc[value_bets['Match_id']==id, f'scoreline'] = result_dict['scoreline']
                value_bets.loc[value_bets['Match_id']==id, f'{bet_name}_bet'] = bet
                value_bets.loc[value_bets['Match_id']==id, f'{bet_name}_portion'] = portion
                self.pay_bet(bet=bet, bet_name=bet_name)
                value_bets.loc[value_bets['Match_id']==id, f'{bet_name}_result'] = str(result_dict[bet_name])
                self.get_payed_if_won(bet=bet, bet_name=bet_name, result=result_dict[bet_name], bookmaker_odds=bookmaker_odds)
                value_bets.loc[value_bets['Match_id']==id,f'{bet_name}_bank'] = self.current_bankroll[f'{bet_name}_bank']
                value_bets.loc[value_bets['Match_id']==id,f'{bet_name}_ROI'] = self.ROI[f'{bet_name}_roi']
                value_bets.loc[value_bets['Match_id']==id,f'{bet_name}_NetGain'] = self.NetGain[f'{bet_name}_gain']
        
        self.value_bets = value_bets
        return {'Investment': self.starting_bank, 'NetGain': self.NetGain, 'ROI': self.ROI}
        
    def produce_report_figures(self, validation_season: str, evaluation_output: str) -> dict:
        """Produces and saves the reporting figures in the specified path

        Args:
            validation_season (str): The season that is evaluated, one of ['2017', '2018', '2019', '2020', '2021', '2022', '2023']
            evaluation_output (str): The specified path to save the evaluation figures

        Returns:
            dict: A dictionary containing with each key value containing the figure that corresponds to the betting of the certain betting category
        """
        logger.info('Producing report figures...')
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        self.value_bets['CombinedLabel'] = self.value_bets['HomeTeam'] + '-' + self.value_bets['AwayTeam'] + ' | ' + self.value_bets['Date'].astype(str)
        figure_dict = {}
        for bet, bookmaker_column in tqdm(zip(['home_win', 'draw', 'away_win', 'over2.5', 'under2.5'], ['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverOdds', 'UnderOdds']), total=5):
            value_bets_only_played = self.value_bets[self.value_bets[f'{bet}_bet'] != 0].copy()
            fig, ax1 = plt.subplots(figsize=(40, 10))
            fig.suptitle(f'{bet} {validation_season} \n Investment: {self.starting_bank}€ | ROI: {round(self.ROI[f"{bet}_roi"], 2)}% Net Gain: {round(self.NetGain[f"{bet}_gain"], 2)}€')
    
            # Plot ROI on the first y-axis (ax1)
            ax1.set_xlabel('Teams | Date')
            ax1.set_ylabel('ROI', color='tab:blue')
            sns.lineplot(data=value_bets_only_played, x='CombinedLabel', y=f'{bet}_ROI', ax=ax1, color='tab:blue', marker='o', errorbar=None)
            ax1.tick_params(axis='y', labelcolor='tab:blue')
    
            # Create a second y-axis sharing the same x-axis
            ax2 = ax1.twinx()
            ax2.set_ylabel('Net Gain', color='tab:orange')
            sns.lineplot(data=value_bets_only_played, x='CombinedLabel', y=f'{bet}_NetGain', ax=ax2, color='tab:orange', marker='o', errorbar=None)
            ax2.tick_params(axis='y', labelcolor='tab:orange')
    
            # Find the minimum and maximum values of both y-axes
            y_min, y_max = min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1], 0)
    
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
    
            # Set the x-axis labels and rotate them
            x_labels = value_bets_only_played['CombinedLabel'].tolist()
            x_labels_step = 1  # Adjust the step to control label spacing
            ax1.set_xticks(range(0, len(x_labels), x_labels_step))
            ax1.set_xticklabels(x_labels[::x_labels_step], rotation=90)
    
            # Add text labels at specific x-axis ticks using ax1.annotate()
            for x_tick, id in zip(x_labels[::x_labels_step], value_bets_only_played['Match_id'].tolist()):
                k = x_labels.index(x_tick)  # Get the index of the x_tick label in the DataFrame
                odds_value = self.bookmaker_probabilities.loc[self.bookmaker_probabilities['Match_id']==id, bookmaker_column].values[0]
                result_value = self.results.loc[self.results['Match_id']==id, 'Result'].values[0]
    
                label = f"{odds_value} | {result_value}"
                # Calculate the y-coordinate for the label
                # You can adjust the value (e.g., +0.02) to control the vertical placement
                y_coord = max(y_min, y_max) + 0.02

                # Add the label at the specified x and y coordinates
                ax1.text(x_tick, y_coord, label, rotation=90, ha='center', va='center')

            # Show the legend
            ax1.legend(['ROI'], loc='lower left')
            ax2.legend(['Net Gain'], loc='lower right')

            # Customize the line appearance
            ax1.axhline(y=-self.starting_bank, color='red', linestyle='--', label='Empty Bankroll', linewidth=2, alpha=0.5)
            # Save the figure instead of showing it
            directory_path = os.path.join(evaluation_output, validation_season)
            os.makedirs(directory_path, exist_ok=True)
            plt.tight_layout()
            fig.savefig(os.path.join(directory_path, f'{bet}_figure.png'))
            figure_dict[f'{bet}_figure'] = fig
            plt.close(fig)  # Close the figure to free up resources
        return figure_dict

            
