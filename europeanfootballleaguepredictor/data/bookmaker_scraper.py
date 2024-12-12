from bs4 import BeautifulSoup
import time
from datetime import datetime
import pandas as pd
from loguru import logger
import undetected_chromedriver as uc
import uuid
import os 
import certifi


def combine_dictionaries(dictionary_list: list):
    combined_list = []
    for odds_type_dict in dictionary_list:
        for dic in odds_type_dict:
            identifier = dic["id"]
            if not has_dict_with_id(combined_list, identifier):
                combined_list.append(dic)
            else:
                for match in combined_list:
                    if match['id']==identifier:
                        match.update(dic)
                        pass
                    
    return combined_list
    
def has_dict_with_id(lst, target_id):
    if not lst:  # Check if the list is empty
        return False
    for d in lst:
        if 'id' in d and d['id'] == target_id:
            return True
    return False
    
class BookmakerScraper():
    """A class responsible for scraping the bookmaker website"""
    def __init__(self, url: str, dictionary: dict):
        """Initializing the scraper by specifying the url and the dictionary of the team names used by the bookmaker.

        Args:
            url (str): The url corresponding to the certain webpage with the betting odds of the league specified in the configuration.
            dictionary (dict): A dictionary of the team names used by the bookmaker.
        """
        self.result_url, self.over_under_url, self.btts_url = BookmakerScraper.produce_urls(url)
        self.dictionary = dictionary
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
        os.environ["SSL_CERT_FILE"] = certifi.where()
        self.driver = uc.Chrome(version_main = 131)

    @staticmethod    
    def produce_urls(base_url: str):
        """
        Transforms the base URL of match result to over/under and btts urls.
        """
        result_url = base_url + '?bt=matchresult'
        over_under_url = base_url + '?bt=overunder'
        btts_url = base_url + '?bt=bothteamstoscore'
        
        return result_url, over_under_url, btts_url
    
    def scroll_down(self):
        """A method for scrolling the page."""

        # Get scroll height.
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:

            # Scroll down to the bottom.
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load the page.
            time.sleep(2)

            # Calculate new scroll height and compare with last scroll height.
            new_height = self.driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height:

                break

            last_height = new_height
    
    def get_page_soup(self, url):
        self.driver.get(url)
        self.driver.implicitly_wait(4)
        self.scroll_down()
        self.driver.implicitly_wait(4)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        return soup
    
    def extract_odds(self, soup):
        dictionary_list = []
        all_matches = soup.find('div', {"class":"league-page"}).find('div', {"class":"vue-recycle-scroller__item-wrapper"}).find_all('div', {"class":["vue-recycle-scroller__item-view", "vue-recycle-scroller__item-view hover"]})
        year = datetime.now().year
        for match in all_matches:
            #date = match.find('span', {"class":"tw-mr-0"}).text + '/' + str(year)
            teams = match.find_all('div', class_='tw-truncate')
            home_team = teams[0].text.strip()
            away_team = teams[1].text.strip()
            selections = match.find_all('div', "selections__selection")
            odds_names = []
            odds_values = []
            for selection in selections:
                odds_names.extend([odd.text for odd in selection.find('span', class_='selection-horizontal-button__title')])
                odds_values.extend([odd.text for odd in selection.find('span', class_='tw-text-s')])
            
            line_values = ['0.5', '1.5', '2.5', '3.5', '4.5', '5.5', '6.5']
            match_dictionary = {'HomeTeam': home_team, 'AwayTeam': away_team}

            for title in odds_names:
                if title in line_values:
                    match_dictionary['Line'] = title
                    odds_names.remove(title)
                    break 

            for title, value in zip(odds_names, odds_values):
                if 'Over' in title:
                    title = 'OverLine'
                if 'Under' in title:
                    title = 'UnderLine'
                
                match_dictionary[title] = value
            dictionary_list.append(match_dictionary)
        
        identified_list = self.generate_uuids(dictionary_list, ['HomeTeam', 'AwayTeam'])   

        return identified_list
    
    @staticmethod
    def generate_uuids(list_of_dicts, keys):
        for i, data in enumerate(list_of_dicts):
            key_values = '-'.join([str(data[key]) for key in keys])
            id = uuid.uuid5(uuid.NAMESPACE_OID, key_values)
            data['id'] = str(id)
            list_of_dicts[i] = data
            
        return list_of_dicts
    
    def return_odds(self):
        odds_json_list = []
        for url in [self.result_url, self.over_under_url, self.btts_url]:
            soup = self.get_page_soup(url)
            extracted_odds_dict = self.extract_odds(soup)
            odds_json_list.append(extracted_odds_dict)
            
        combined_odds = combine_dictionaries(odds_json_list)
        odds_dataframe = pd.DataFrame(combined_odds).drop(columns=['id'])
        translated_names_odds = self.replace_team_names(odds_dataframe, self.dictionary)
        self.driver.quit()
        return translated_names_odds
    
    def replace_team_names(self, input_dataframe: pd.DataFrame, replacing_dict: dict) -> pd.DataFrame:
        """
        Replaces the team names used by bookmaker with the format of understat.

        Args:
            input_dataframe (pd.DataFrame): The scraped odds dataframe.
            replacing_dict (dict): A team name mapping to communicate different team names between bookmaker and understat.
        """
        input_dataframe['HomeTeam'] = input_dataframe['HomeTeam'].replace(replacing_dict, regex=False)
        input_dataframe['AwayTeam'] = input_dataframe['AwayTeam'].replace(replacing_dict, regex=False)
        
        return input_dataframe
            
    
