import requests
from bs4 import BeautifulSoup
import json
import re
import datetime
import pandas as pd
from loguru import logger

class BookmakerScraper():
    """A class responsible for scraping the bookmaker website
    """
    def __init__(self, url: str, dictionary: dict):
        """Initializing the scraper by specifying the url and the dictionary of the team names used by the bookmaker

        Args:
            url (str): The url corresponding to the certain webpage with the betting odds of the league specified in the configuration
            dictionary (dict): A dictionary of the team names used by the bookmaker
        """
        self.url = url
        self.dictionary = dictionary
    
    def get_odds_json(self) -> dict:
        """Gets a page dictionary containing the odds, from the specified url

        Returns:
            dict: A dictionary containing the odds together with other raw code elements
        """
        page = requests.get(self.url)
        soup = BeautifulSoup(page.content, "html.parser")
        script = soup.body.find_all("script")
        script = script[0]

        # Define the regular expression pattern
        pattern = r'<script>(.*?)<\/script>'

        # Use re.findall to find all matches
        matches = re.findall(pattern, str(script), re.DOTALL)

        if matches:
            # Extract the content between script tags
            content_between_scripts = matches[0]
            content_between_scripts= content_between_scripts.split('window["initial_state"]=')[1].strip()
        else:
            print('No match found.')

        odds_dictionary = json.loads(content_between_scripts)
        
        return odds_dictionary
    
    def odds_json_to_dataframe(self, odds_dictionary: dict) -> pd.DataFrame:
        """Produces a dataframe out of the dictionary output by get_odds_json()

        Args:
            odds_dictionary (dict): A dictionary containing the odds together with other raw code elements, output of get_odds_json()

        Returns:
            pd.DataFrame: A dataframe containing the bookmaker odds for the upcoming matches of the specified league
        """
        Odds = []
        Dates = []
        HomeTeams = []
        AwayTeams = []
        rows = []
        names = []
        for match in odds_dictionary['data']['blocks'][0]['events']:
            teams = match['shortName'].split(' - ')
            HomeTeams.append(self.replace_names(teams[0]))
            AwayTeams.append(self.replace_names(teams[1]))
            Dates.append(datetime.datetime.fromtimestamp(match['startTime']/1000.0).strftime('%d/%m/%Y'))
            for x in match['markets']:
                for y in x['selections']:
                    names.append(y['name'])
                    Odds.append(y['price'])
            
            while Odds:
                line = names[4].split(' ')[1]
                rows.append(Odds[:3] + [line] + Odds[3:7])
                Odds = Odds[7:]
                names = names[7:]
                
        data_teams = pd.DataFrame({'Home Team': HomeTeams, 'Away Team': AwayTeams})
        logger.debug(line)
        data_values = pd.DataFrame(rows, columns=['1', 'x', '2', 'Line', 'OverLineOdds', 'UnderLineOdds', 'Yes', 'No'])
        odds_dataframe = pd.concat([data_teams, data_values], axis=1)
        odds_dataframe['Home Team'] = odds_dataframe['Home Team'].str.strip()
        odds_dataframe['Away Team'] = odds_dataframe['Away Team'].str.strip()
        
        return odds_dataframe
    
    @logger.catch
    def get_odds(self) -> pd.DataFrame:
        """A pipeline that directly provides the odds_dataframe

        Returns:
            pd.DataFrame: A dataframe containing the bookmaker odds for the upcoming matches of the specified league
        """
        odds_json = self.get_odds_json()
        odds_dataframe = self.odds_json_to_dataframe(odds_json)
        return odds_dataframe
         
    def replace_names(self, string) -> str:
        """Replaces team names using the dictionary in order to match the team names used by understat

        Args:
            string (str): The string that should be examined and get team names updated

        Returns:
            str: The input string with the team names now corresponding to understat format
        """
        for old_name, new_name in self.dictionary.items():
            string = string.replace(old_name, new_name)
        return string
    

