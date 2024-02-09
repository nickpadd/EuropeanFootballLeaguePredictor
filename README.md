# EuropeanFotballLeaguePredictor

EuropeanFootballLeaguePredictor is a predictor of europes top league games based on historic performances of teams, taking into account only advanced league table statistics. It uses traditional machine learning techniques and statistical modeling in order to deduct data driven predictions based on the team’s season performance and recent form. The EuropeanFootballLeaguePredictor includes EnglishPremierLegue, La Liga, Ligue 1, Bundesliga, Serie A. 

## Betting Disclaimer

The EuropeanFootballLeaguePredictor project and its associated predictions are provided for informational purposes. The predictions generated by this project should not be considered as financial advice or recommendations to place bets on any Premier League matches or other events.

Betting involves risks, and there is no guarantee that the predictions provided by this project will result in accurate outcomes or profitable bets. The outcomes of sports events can be influenced by a wide range of variables that may not be fully captured by the prediction model.


## Visit the Github Page
The [EuropeanFootballLeaguePredictor Github Page](https://nickpadd.github.io/EFLP.github.io/Home) provides a detailed description of the model creation and evaluation on past seasons, as well as the upcoming predictions of this week's european league games!


## How to run the project
### Installing Project Dependencies
1. Clone this repository to your local machine or download the project files.

   ```bash
   git clone https://github.com/nickpadd/EuropeanFootballLeaguePredictor

2. Navigate to the project directory using the command line.
    ```bash
    cd EuropeanFootballLeaguePredictor

- Managing dependencies with Poetry: 
    - Installing poetry:
    ```bash
    pip install poetry 
    ```
    - Installing project dependencies:
    ```bash
    poetry install
    ```

    - Running the poetry shell (a virtual environment with the dependencies installed):
    ```bash
    poetry shell

- Alternative method via venv:
    - Create a virtual environment using venv:
    ```bash
    python3 -m venv venv
    ```
    - Activate the virtual environment:
        - On Windows:
        ```bash
        venv\Scripts\activate
        ```
        - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    - Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration
The configuration file europeanfootballleaguepredictor/config/config.yaml includes many of the models parameters such as:

- **league**: The league for which the scripts are run.
- **model.regressor**: Able to use one of LinearRegression, PoissonRegressor, SVR
- **model.votes.long_term_form_vote_perc/ model.votes.short_term_form_vote_perc**: The voting weights of the long form statistics and short term form statistics. Should add to 1.
- **model.matchdays_to_drop**: The matchdays at the start of the season the model should ignore as they are considered to include redundant statistics.
- **data_gathering.long_term_form/ data_gathering.short_term_form**: The months of form that are considered long and short term form. Null is parsed as season long form. 
- **bettor.initial_bank**: The initial investment which the bettor uses in each distinct betting category during the evaluation.
- **bettor.kelly_cap**: The maximum percentage of the current bankroll to bet on using the kelly betting criterion.


### Running the scripts for the upcoming matches
For running the scripts below make sure you have followed [Installing Project Dependencies](#installing-project-dependencies) and you are inside the poetry shell.

- Data Collection: </br>
In order to run the model running the 'run_data_collection.py' is required. The script uses the configuration settings **data_gathering.long_term_form/ data_gathering.short_term_form** in order to gather the requested data.
```bash
python run_data_collection.py
```
This will initialize the database of the league specified in the configuration and gather the requested data of long term and short term form.

- Updates: </br>
To update the database for recent game results and upcoming fixtures:
```bash
python run_updates.py
```

- Predictions: </br>
The predictions can be run either directly by a python script or via a web-application.

    - To start the web-application run the following line. A localhost url will appear which hosts the application:
    ```bash
    python predictor_app.py
    ```
    - To run directly make sure you have specified the configuration settings for the league and regressor model and matchdays to drop and run the following line. This will produce a Predictions/{league} folder where the predictions are saved as an interactive plot and a html table:
    ```bash
    python run_predictions.py
    ```

- Evaluation: </br>
The evaluation script is run on the specified league and parameters of the configuration settings with the following line:

```bash
python run_evaluation_per_season.py
```

Leave one season out technique is used excluding the 2023 season which is left out for the final model test. For each run the model will train on all seasons but one and bet on the left out season and producing important metrics and figures that are logged to mlflow. The figures can also be found in 'europeanfootballleague/data/leagues/{league}/evaluation/' directory. </br>
Each figure is the ROI and Net-Gain curve for all the bets that the model produced in the exact betting category.
The metrics are produced for each year independently and consist of:


- **ROI_12**: The mean ROI for home_win and away_win betting
- **ROI_1x2**: The mean ROI for home_win, draw, away_win betting
- **ROI_ou**: The mean ROI for over2.5/under2.5 goal betting
- **ROI_home_win**: The ROI for the home_win betting model
- **ROI_draw**: The ROI for the draw betting model
- **ROI_away_win**: The ROI for the away_win betting model
- **ROI_over2.5**: The ROI for the over2.5 goal betting model
- **ROI_under2.5**: The ROI for the under2.5 goal betting model

In order to access the logged runs use the mlflow user interface by running:
```bash
mlflow ui
```
This will host a user interface in local hosts were you will be able to examine and compare model performances.

## Optimal model parameters and performance
Currently ongoing model evaluation of optimal hyperparameters and performance. Will be updated soon.


