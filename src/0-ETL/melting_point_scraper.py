from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import sqlite3
import pubchempy as pcp
import urllib
import time
import re
import logging
import json
from tqdm import tqdm



class MeltingPointScraper:
    def __init__(self):
        self.service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=self.service)
        self.driver.implicitly_wait(10)

    def close(self):
        self.driver.quit()

    def get_melting_points(self, item):
        self.driver.get("https://www.chemspider.com/")
        try:
            consent_button = self.driver.find_element(By.ID, "onetrust-accept-btn-handler")
            consent_button.click()
            self.driver.implicitly_wait(10)
        except:
            pass

        search_bar = self.driver.find_element(By.ID, "input-left-icon")
        search_bar.send_keys(item)
        search_bar.send_keys(Keys.RETURN)
        
        while True:
            self.driver.implicitly_wait(1)
            try:
                self.driver.find_element(By.ID, "tab1").click()
                break
            except:
                
                try:
                    tbody = self.driver.find_element(By.TAG_NAME, "tbody")
                    print("Multiple results found, selecting first one...")
                    first_row = tbody.find_elements(By.TAG_NAME, "tr")[0]
                    first_row.click()
                    break
                except:
                    print("\nNot a multiple result error, trying again...")
                
                
                self.driver.get(self.driver.current_url)
                self.driver.implicitly_wait(1)
                search_bar = self.driver.find_element(By.ID, "input-left-icon")
                search_bar.send_keys(item)
                search_bar.send_keys(Keys.RETURN)
                self.driver.implicitly_wait(1)

        properties_tab = self.driver.find_element(By.ID, "tab1")
        properties_tab.click()
        self.driver.implicitly_wait(1)

        melting_point_link = self.driver.find_element(By.XPATH, "//span[text()='Melting point']")
        melting_point_link.click()
        self.driver.implicitly_wait(1)

        melting_container = self.driver.find_element(By.ID, "accordion-melting-point")
        table_body = melting_container.find_element(By.TAG_NAME, "tbody")
        rows = table_body.find_elements(By.TAG_NAME, "tr")

        melting_points = []
        for row in rows:
            columns = row.find_elements(By.TAG_NAME, "td")
            melting_points.append({'label': columns[1].text, 'value': columns[0].text})

        return melting_points

def get_all_ids():
    conn = sqlite3.connect('db/MasterDatabase.db')
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM compounds")
    compound_ids = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT id FROM solvents")
    solvent_ids = [row[0] for row in cursor.fetchall()]

    conn.close()
    return compound_ids, solvent_ids

def get_cas(pub_chem_id: int) -> dict:
    while True:
        try:
            compound = pcp.Compound.from_cid(pub_chem_id) 
            synonyms = compound.synonyms
            for synonym in synonyms:
                if re.match(r'^\d{2,7}-\d{2}-\d{1}$', synonym):
                    return synonym
            return None
        except (pcp.PubChemHTTPError, urllib.error.URLError) as e:
            if 'SSL' in str(e):
                logging.warning(f"SSL error for PubChem ID {pub_chem_id}, retrying...")
                continue
            else:
                raise



compound_ids, solvent_ids = get_all_ids()

results = {'solvents': [], 'compounds': []}

for id in tqdm(compound_ids, desc="Processing compound IDs"):
    cas = get_cas(id)
    results['compounds'].append({'ID': id, 'CAS': cas})
    
for id in tqdm(solvent_ids, desc="Processing solvent IDs"):
    cas = get_cas(id)
    results['solvents'].append({'ID': id, 'CAS': cas})

with open('db/results.json', 'w') as f:
    json.dump(results, f, indent=4)


with open('db/results.json', 'r') as f:
    results = json.load(f)

scraper = MeltingPointScraper()
for compound in tqdm(results['compounds'], desc="Processing compounds"):
    cas = compound['CAS']
    if cas:
        try:
            melting_points = scraper.get_melting_points(cas)
            print(melting_points)
            compound['melting_points'] = melting_points
        except Exception as e:
            logging.error(f"Failed to get melting points for compound {compound['ID']} with CAS {cas}: {e}")
            compound['melting_points'] = None

for solvent in tqdm(results['solvents'], desc="Processing solvents"):
    cas = solvent['CAS']
    if cas:
        try:
            melting_points = scraper.get_melting_points(cas)
            print(melting_points)
            solvent['melting_points'] = melting_points
        except Exception as e:
            logging.error(f"Failed to get melting points for solvent {solvent['ID']} with CAS {cas}: {e}")
            solvent['melting_points'] = None

scraper.close()

with open('assets/features/MeltingPoints.json', 'w') as f:
    json.dump(results, f, indent=4)


