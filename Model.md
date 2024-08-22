# The Model
Given the inherent diversity within football and the remarkable advancements in data-driven technologies over recent years, the selection of training data for this project carries significant weight and should be approached with careful consideration. The model is exclusively trained on the same leagues football matches spanning from 2017 to 2022 in order to balance between the quantity and quality of the data given the progress of the sport in the past years and the difference in other football leagues.

## Data Collection and Training

The historic data were collected from the league table using [Understat](https://understat.com/ "Understat's Homepage") and its [API](https://understat.readthedocs.io/en/latest/ "Understat API") from 2017 to 2022 for each team, one day before each individual match. This data was then correlated with historic results and odds gathered through [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php "Football-Data.co.uk"). A regression algorithm was trained to predict Home Goals and Away Goals which were then used to model each team's goal scoring using the Poisson probability density function, resulting in a probability estimation for each distinct scoreline.

## Including Form into the Method

To incorporate recent form into our predictions, the above process was repeated, but only considering the last three months of the league table. This allowed us to compute predictions based on the form of teams for the past games before each match. A voting system between the season-long prediction and the form-based