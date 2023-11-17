import numpy as np 
from loguru import logger 
import matplotlib.pyplot as plt
import seaborn as sns 
import os
from tqdm import tqdm 
from sklearn.svm import SVC

class Bettor:
    def __init__(self, bank: float, margin_dictionary: dict):
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
        self.margin_dictionary = margin_dictionary
        self.result_dict = {}
    
    def reset_bank(self):
        self.__init__(bank= self.starting_bank, margin_dictionary=self.margin_dictionary)
        
    def preprocess(self, prediction_dataframe, results):    
        info_columns = ['Match_id', 'Date', 'HomeTeam', 'AwayTeam']
        bookmaker_columns =['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverOdds', 'UnderOdds']
        model_columns = ['HomeWinProbability', 'DrawProbability', 'AwayWinProbability', 'Over2.5Probability', 'Under2.5Probability']
        self.info = prediction_dataframe[info_columns]
        self.bookmaker_probabilities = prediction_dataframe[bookmaker_columns+['Match_id']]
        self.model_probabilities = prediction_dataframe[model_columns+['Match_id']]
        self.results = results.reset_index(drop=True)
        
    def check_dictionary_compliance(self, bookmaker_odds: float, estimated_true_probability: float, bet_name):
        if estimated_true_probability - 1/bookmaker_odds > self.margin_dictionary[bet_name]:
            return True 
        else: 
            return False
        
    def kelly_criterion(self, bookmaker_odds: float, estimated_true_probability: float, bet_name):
        dict_allows_bet = self.check_dictionary_compliance(bookmaker_odds= bookmaker_odds, estimated_true_probability= estimated_true_probability, bet_name= bet_name)
        if dict_allows_bet:
            portion_of_bet_gained_with_win = bookmaker_odds - 1
            bankroll_portion = estimated_true_probability - (1 - estimated_true_probability)/portion_of_bet_gained_with_win
            bet = bankroll_portion*self.current_bankroll[f'{bet_name}_bank']
            return bankroll_portion, max(bet, 0)/4
        else:
            return 0, 0
    
    def get_betting_result(self, match_id):
        scoreline = self.results.loc[self.results['Match_id']==match_id, 'Result'].values[0]
        home_goals = int(scoreline.split('-')[0])
        away_goals = int(scoreline.split('-')[1])
    
        result_dict = {
            'home_win': True if home_goals>away_goals else False,
            'draw': True if home_goals==away_goals else False,
            'away_win': True if home_goals<away_goals else False,
            'over2.5': True if home_goals+away_goals>=3 else False,
            'under2.5': True if home_goals+away_goals<3 else False
        }
        self.result_dict[match_id] = result_dict
        return result_dict
    
    def pay_bet(self, bet, bet_name):
        self.current_bankroll[f'{bet_name}_bank'] -= bet
    
    def get_payed_if_won(self, bet, bet_name, result, bookmaker_odds):
        if result == True:
            self.current_bankroll[f'{bet_name}_bank'] += bet*bookmaker_odds
        elif result == False:
            pass
        
        self.NetGain[f'{bet_name}_gain'] = self.current_bankroll[f'{bet_name}_bank'] - self.starting_bank
        self.ROI[f'{bet_name}_roi'] = 100*self.NetGain[f'{bet_name}_gain']/self.starting_bank
            
    def place_value_bets(self):
        logger.info('Betting on the season...')
        bet_columns = ['home_win', 'draw', 'away_win', 'over2.5', 'under2.5']
        value_bets = self.info.copy()
        value_bets[bet_columns] = np.nan
        for index, id in tqdm(enumerate(self.info['Match_id']), total=len(self.info['Match_id'])):
            result_dict = self.get_betting_result(match_id = id)
            for bookmaker_odds_column, model_probability_column, bet_name in zip(self.bookmaker_probabilities.columns, self.model_probabilities.columns, bet_columns):
                bookmaker_odds = self.bookmaker_probabilities.loc[self.bookmaker_probabilities['Match_id']==id, bookmaker_odds_column].values[0]
                model_probability = self.model_probabilities.loc[self.model_probabilities['Match_id']==id, model_probability_column].values[0]
                portion, bet = self.kelly_criterion(bookmaker_odds=bookmaker_odds, estimated_true_probability=model_probability, bet_name=bet_name)
                value_bets.loc[index, 'Match_id'] = id
                value_bets.loc[index, f'{bet_name}_bet'] = bet
                value_bets.loc[index, f'{bet_name}_portion'] = portion
                self.pay_bet(bet=bet, bet_name=bet_name)
                value_bets.loc[index, f'{bet_name}_result'] = str(result_dict[bet_name])
                self.get_payed_if_won(bet=bet, bet_name=bet_name, result=result_dict[bet_name], bookmaker_odds=bookmaker_odds)
                value_bets.loc[index,f'{bet_name}_bank'] = self.current_bankroll[f'{bet_name}_bank']
                value_bets.loc[index,f'{bet_name}_ROI'] = self.ROI[f'{bet_name}_roi']
                value_bets.loc[index,f'{bet_name}_NetGain'] = self.NetGain[f'{bet_name}_gain']
        
        self.value_bets = value_bets
        return {'Investment': self.starting_bank, 'NetGain': self.NetGain, 'ROI': self.ROI}
        
    def produce_report_figures(self, validation_season, evaluation_output):
        logger.info('Producing report figures...')
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        self.value_bets['CombinedLabel'] = self.value_bets['HomeTeam'] + '-' + self.value_bets['AwayTeam'] + ' | ' + self.value_bets['Date'].astype(str)
        figure_dict = {}
        for bet, bookmaker_column in tqdm(zip(['home_win', 'draw', 'away_win', 'over2.5', 'under2.5'], ['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverOdds', 'UnderOdds']), total=5):
            value_bets_only_played = self.value_bets[self.value_bets[f'{bet}_bet'] != 0].copy()
            fig, ax1 = plt.subplots(figsize=(20, 8))
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
            for x_tick in x_labels[::x_labels_step]:
                k = x_labels.index(x_tick)  # Get the index of the x_tick label in the DataFrame
                odds_value = self.bookmaker_probabilities.loc[k, bookmaker_column]
                result_value = self.results['Result'].iloc[k]
    
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
            fig.savefig(os.path.join(directory_path, f'{bet}_figure.png'))
            figure_dict[f'{bet}_figure'] = fig
            plt.close(fig)  # Close the figure to free up resources
        return figure_dict

            
