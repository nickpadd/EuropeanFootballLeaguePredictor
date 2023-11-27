import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np 
from loguru import logger 

class Visualizer:
    """The visualizer of the predicted match probabilities
    """
    def __init__(self, prediction_dataframe: pd.DataFrame) -> None:
        """Initializing the visualizer using the prediction dataframe

        Args:
            prediction_dataframe (pd.DataFrame): A dataframe containing match information and predicted probabilities from the model
        """
        self.prediction_dataframe = prediction_dataframe
        self.get_bookmaker_prob()

    def get_bookmaker_prob(self):
        """Changes the bookmaker odds into probabilities
        """
        try:
            self.prediction_dataframe[['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverLineOdds', 'UnderLineOdds', 'Yes', 'No']] = 1/self.prediction_dataframe[['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverLineOdds', 'UnderLineOdds', 'Yes', 'No']]
        except KeyError:
            logger.warning('Bookmaker results were not found, will produce prediction figures not containing that data.')
    
    def produce_closest_line(self):
        # Extracting the numerical values from the column names
        lines = ['1.5', '2.5', '3.5']

        # Calculate absolute differences between "OverX" and "UnderX" for each X
        differences = {line: np.abs(self.prediction_dataframe[f'Over{line}Probability'] - self.prediction_dataframe[f'Under{line}Probability']) for line in lines}

        # Determine the line with the minimum absolute difference for each row
        self.prediction_dataframe['Line'] = [min(differences.keys(), key=lambda x: differences[x][i]) for i in range(len(self.prediction_dataframe))]

        # Convert the "Line" column to string
        self.prediction_dataframe['Line'] = self.prediction_dataframe['Line'].astype(str)
            
    def radar_scoreline_plot(self):
        """Creates the figure containing the scoreline barplot and the radar probability plot

        Returns:
            matplotlib.figure.Figure: A figure containing the scoreline barplot and the radar probability plot
        """
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}, {'type': 'xy'}]], column_widths=[0.33, 0.67])
        fig.update_layout(width=1500, height=650)
        # create a list to store the buttons
        buttons = []

        # loop through each match and create two traces for Bookmaker odds and predicted values
        for index, (h_team, a_team) in enumerate(zip(self.prediction_dataframe['HomeTeam'].tolist(), self.prediction_dataframe['AwayTeam'].tolist())):
        
            # create the trace for Bookmaker odds, try except block incase bookmaker odds are not included
            try:
                trace_bookmaker = go.Scatterpolar(
                  r=100*np.round(np.array(self.prediction_dataframe.iloc[index][['HomeWinOdds', 'DrawOdds', 'AwayWinOdds', 'OverLineOdds', 'UnderLineOdds', 'Yes', 'No']].values.tolist()), 4),
                  theta=['HomeWin', 'Draw', 'AwayWin', f"Over{self.prediction_dataframe.iloc[index]['Line']}", f"Under{self.prediction_dataframe.iloc[index]['Line']}", 'GG', 'NG'],
                  fill='toself',
                  name='Bookmaker Odds',
                  marker = dict(color = 'rgb(82, 106, 131)' ),
                  visible=False if index!=0 else True,
                  hovertemplate='%{theta}<br>Probability: %{r:.2f}%<br>Bookmaker Probability'
                )
                fig.add_trace(trace_bookmaker, row=1, col=1)
                mode = 'odds'
            except KeyError:
                #Produces the over under line with the closest margin
                self.produce_closest_line()
                mode = 'no_odds'
            
            # create the trace for predicted values
            trace_predicted = go.Scatterpolar(
                r= 100*np.round(np.array(self.prediction_dataframe.iloc[index][['HomeWinProbability', 'DrawProbability', 'AwayWinProbability', f"Over{self.prediction_dataframe.iloc[index]['Line']}Probability", f"Under{self.prediction_dataframe.iloc[index]['Line']}Probability", 'GGProbability', 'NGProbability']].values.tolist()), 4),
                theta=['HomeWin', 'Draw', 'AwayWin', f"Over{self.prediction_dataframe.iloc[index]['Line']}", f"Under{self.prediction_dataframe.iloc[index]['Line']}", 'GG', 'NG'],
                fill='toself',
                name='Predicted Odds',
                marker = dict(color ='rgb(141,211,199)'), #'rgb(217, 175, 107)'
                visible=False if index!=0 else True,
                hovertemplate='%{theta}<br>Probability: %{r:.2f}%<br>Predicted Probability'
          )
            fig.add_trace(trace_predicted, row=1, col=1)

            scoreline_dict = self.prediction_dataframe.iloc[index]['ScorelineProbability'][0]
            keys = list(scoreline_dict.keys())
            values = list(scoreline_dict.values())

            scoreline_frame = pd.DataFrame(values, columns=['P(score)'])
            scoreline_frame['score'] = keys
            scoreline_frame['display_score'] = scoreline_frame['score']  # New column for display in hover text
            scoreline_frame.loc[scoreline_frame["P(score)"] < 0.02, "display_score"] = 'Other'
            scoreline_frame = scoreline_frame.sort_values(by=['P(score)'], ascending=False)

            # Rest of your code
            trace_goals = go.Bar(
                x=scoreline_frame['display_score'],
                y=100 * np.round(scoreline_frame['P(score)'], 4),
                visible=False if index != 0 else True,
                marker=dict(color=scoreline_frame['P(score)'], colorscale='darkmint'),
                name='Probable Scoreline',
                hovertemplate='Score %{customdata}<br>Probability: %{y:.2f}%',
                customdata=scoreline_frame['score']  # Use the original score for hover text
            )
            fig.add_trace(trace_goals, row=1, col=2)
             
            # create the dictionary for the button
            if mode=='odds':
                button_dict = dict(
                    method="restyle",
                    args=[{"visible": [False] * (3 * len(self.prediction_dataframe))}], # set all traces to false initially
                    label= f"<b>{h_team}-{a_team} | {str(self.prediction_dataframe.loc[index, 'Date'].strip('[]'))}</b>" 
                )
                # set visibility to true for the corresponding traces
                button_dict["args"][0]["visible"][3*index] = True # Bookmaker odds trace
                button_dict["args"][0]["visible"][3*index+1] = True # Predicted trace
                button_dict["args"][0]["visible"][3*index+2] = True # Goal trace

                # set visibility to true for the corresponding traces for the first match
                if index == 0:
                    button_dict["args"][0]["visible"][0] = True # Bookmaker odds trace
                    button_dict["args"][0]["visible"][1] = True # Predicted trace
                    button_dict["args"][0]["visible"][2] = True

            if mode=='no_odds':
                button_dict = dict(
                    method="restyle",
                    args=[{"visible": [False] * (2 * len(self.prediction_dataframe))}], # set all traces to false initially
                    label= f"<b>{h_team}-{a_team} | {str(self.prediction_dataframe.loc[index, 'Date'].strip('[]'))}</b>" 
                )
                # set visibility to true for the corresponding traces
                button_dict["args"][0]["visible"][2*index] = True # Predicted trace
                button_dict["args"][0]["visible"][2*index+1] = True # Goal trace

                # set visibility to true for the corresponding traces for the first match
                if index == 0:
                    button_dict["args"][0]["visible"][0] = True # Predicted trace
                    button_dict["args"][0]["visible"][1] = True # Goals trace

            
            # append the button to the list
            buttons.append(button_dict)

        # update the layout with the buttons
        fig.update_layout(
            updatemenus=[go.layout.Updatemenu(
                buttons=buttons,
                active=0, # set the initial active button
                x=0.1,
                y=1.2,
                direction="down",
            )],

        )

        return fig
