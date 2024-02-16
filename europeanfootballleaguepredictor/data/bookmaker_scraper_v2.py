from bs4 import BeautifulSoup
import time
from datetime import datetime
import pandas as pd
from loguru import logger
from selenium import webdriver 
from selenium.webdriver.common.proxy import Proxy, ProxyType
import undetected_chromedriver as uc
import uuid


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
        self.driver = uc.Chrome(version_main = 120)

    @staticmethod    
    def produce_urls(base_url: str):
        """
        Transforms the base URL of match result to over/under and btts urls.
        """
        result_url = base_url + '?bt=matchresult'
        over_under_url = base_url + '?bt=overunder'
        btts_url = base_url + '?bt=bothteamstoscore'
        
        return result_url, over_under_url, btts_url
    
    def get_page_soup(self, url):
        self.driver.get(url)
        self.driver.implicitly_wait(4)
        # Scroll down using JavaScript
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        self.driver.implicitly_wait(4)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        return soup
    
    def extract_odds(self, soup):
        dictionary_list = []
        all_matches = soup.find('div', {"class":"league-page"}).find('div', {"class":"vue-recycle-scroller__item-wrapper"}).find_all('div', {"class":"vue-recycle-scroller__item-view"})
        year = datetime.now().year
        for match in all_matches:
            date = match.find('span', {"class":"tw-mr-0"}).text + '/' + str(year)
            teams = match.find_all('span', {"class":"tw-text-n-13-steel tw-inline-block tw-align-top tw-w-auto tw-pl-xs"})
            home_team = teams[0].text.strip()
            away_team = teams[1].text.strip()
            selections = match.find_all('div', "selections")
            for selection in selections:
                odds_names = [odd.text for odd in selection.find_all('span', "selections__selection__title")]
                odds_values = [odd.text for odd in selection.find_all('span', "selections__selection__odd")]
            
            match_dictionary = {'date': date, 'home_team': home_team, 'away_team': away_team}
            for title, value in zip(odds_names, odds_values):
                match_dictionary[title] = value
                
            dictionary_list.append(match_dictionary)
        
        identified_list = self.generate_uuids(dictionary_list, ['date', 'home_team', 'away_team'])   
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
            odds_json_list.append(self.extract_odds(soup)[0])
            
        combined_odds = self.combine_dictionaries(odds_json_list)
        logger.debug(combined_odds)
        self.driver.quit()
        return pd.DataFrame(combined_odds)
            
    @staticmethod
    def combine_dictionaries(dictionary_list: list):
        combined_dict = {}
        for dic in dictionary_list:
            identifier = dic["id"]
            if identifier in combined_dict:
                combined_dict[identifier].update(dic)
            else:
                combined_dict[identifier] = dic
        
        return combined_dict