import gradio as gr
from loguru import logger 
import argparse
from europeanfootballleaguepredictor.common.config_parser import Config_Parser
import pandas as pd
from europeanfootballleaguepredictor.models.probability_estimator import ProbabilityEstimatorNetwork
from europeanfootballleaguepredictor.visualization.visualize import Visualizer
from pretty_html_table import build_table
import importlib
import os
import argparse
from europeanfootballleaguepredictor.data.database_handler import DatabaseHandler 


def league_predictions_figure(league_name, regressor_str, matchdays_to_drop, long_form_vote):
    """Generates a prediction table with the model predictions of certain match outcomes 
    and a prediction figure that includes a radar plot comparing the bookmakers odds depicted as probabilities with the predicted probabilities
    as well as a barplot with the most probable scorelines according to the model. 
    
    Args:
        league_name (str): One of the available leagues ['EPL', 'Bundesliga', 'Ligue_1', 'La_Liga', 'Serie_A']
        regressor_str (str): One of the available regressors ['LinearRegression', 'PoissonRegressor', 'SVR']
        matchdays_to_drop (int): The matchdays at the start of the season that are considered to provide redundant information to the model because the league table is not yet indicative of the teams performance due to small sample size.
        long_form_vote (int): The weight with which the model produces the predictions between long form and short form. The short_form_vote is then calculated by 1-long_form_vote. Long form and short form are dependent on the users configuration before the data collection. Defaults are long_form : season long form, short_form : 3 month form.

    Returns:
        plotly_figure: A prediction figure that includes a radar plot comparing the bookmakers odds depicted as probabilities with the predicted probabilities as well as a barplot with the most probable scorelines according to the model.
        html_table: A prediction table with the model predictions of certain match outcomes.
    """
    
    voting_dict = {'long_term': long_form_vote, 'short_term': 1-long_form_vote} 
    '''Parsing the configuration file'''
    if (regressor_str == "LinearRegression") or (regressor_str == "PoissonRegressor"):
        module_path = 'sklearn.linear_model'
        regressor_module = importlib.import_module(module_path)
        regressor = getattr(regressor_module, regressor_str)
    if regressor_str == 'SVR':
        module_path = 'sklearn.svm'
        regressor_module = importlib.import_module(module_path)
        regressor = getattr(regressor_module, regressor_str)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file (e.g., config.yaml)", default='europeanfootballleaguepredictor/config/config.yaml')
    config_file_path = parser.parse_args().config
    
    config_data_parser = Config_Parser(config_file_path, None)
    config_data = config_data_parser.load_and_extract_yaml_section()
    config = config_data_parser.load_configuration_class(config_data)
    
    logger.info(config)
    '''End of the configuration file parsing'''
    
    pd.set_option('display.precision', 2)
    
    net = ProbabilityEstimatorNetwork(voting_dict=voting_dict, matchdays_to_drop=matchdays_to_drop)
    net.build_network(regressor = regressor)
        
    database = f'europeanfootballleaguepredictor/data/database/{league_name}_database.db'
    database_handler = DatabaseHandler(league=config.league, database=database)
    short_term_form, long_term_form, for_prediction_short, for_prediction_long = database_handler.get_data(table_names=["Preprocessed_ShortTermForm", "Preprocessed_LongTermForm", "Preprocessed_UpcomingShortTerm", "Preprocessed_UpcomingLongTerm"])
    
    probability_dataframe = net.produce_probabilities(short_term_data=short_term_form, long_term_data=long_term_form, for_prediction_short=for_prediction_short, for_prediction_long=for_prediction_long)

    visualizer = Visualizer(probability_dataframe)
    figure = visualizer.radar_scoreline_plot()
    html_table = build_table(probability_dataframe.drop(['ScorelineProbability', 'Match_id'], axis=1), 'blue_light')
    
    return figure, html_table

def main():
    """Main function of the up initializes the gradio app interface with which the user runs the model.
    """
    with gr.Blocks() as iface:
        with gr.Row():
            drop1 = gr.Dropdown(['EPL', 'Bundesliga', 'Ligue_1', 'La_Liga', 'Serie_A'], label="Select League")
            drop3 = gr.Dropdown(['LinearRegression', 'PoissonRegressor', 'SVR'], label="Select regressor Type")
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                slide1 = gr.Slider(minimum=0, maximum=10, step=1, value=4, show_label=True, interactive=True, label="Select number of matchdays to drop")
                slide2 = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.6, show_label=True, interactive=True, label="Select long form vote. Short form will be 1-long")
                btn = gr.Button("Predict")
            with gr.Column(scale=5, min_width=1200):
                plot1 = gr.Plot(label='Predicted results plots')
                table1 = gr.HTML(label='Predicted results tables')

        btn.click(league_predictions_figure, inputs=[drop1, drop3, slide1, slide2], outputs=[plot1, table1])

    iface.launch(height=2000)
    
if __name__ == "__main__":
    main()