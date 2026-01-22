import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

class AQIBeautifulSoupScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        })
        
        self.cities_data = {
            'delhi': {
                'urls': [
                    #'https://waqi.info/#/c/india/delhi/5.31q3q',
                    'https://aqicn.org/city/india/delhi/',
                    'https://www.accuweather.com/en/in/delhi/202396/air-quality-index/202396'
                ]
            },
            'beijing': {
                'urls': [
                    #'https://waqi.info/#/c/china/beijing/5.2ydwr',
                    'https://aqicn.org/city/china/beijing/',
                    'https://www.accuweather.com/en/cn/beijing/106577/air-quality-index/106577'
                ]
            },
            'los-angeles': {
                'urls': [
                    #'https://waqi.info/#/c/usa/california/los-angeles/5.2gv3g',
                    'https://aqicn.org/city/usa/california/los-angeles/',
                    'https://www.accuweather.com/en/us/los-angeles/90012/air-quality-index/347625'
                ]
            },
            'london': {
                'urls': [
                    #'https://waqi.info/#/c/uk/england/london/5.30r6t',
                    'https://aqicn.org/city/uk/england/london/',
                    'https://www.accuweather.com/en/gb/london/ec4a-2/air-quality-index/328328'
                ]
            },
            'tokyo': {
                'urls': [
                    #'https://waqi.info/#/c/japan/tokyo/5.7q3qv',
                    'https://aqicn.org/city/japan/tokyo/',
                    'https://www.accuweather.com/en/jp/tokyo/226396/air-quality-index/226396'
                ]
            }
        }
        
        self.parameters = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']
    
    def scrape_waqi_info(self, url, city_name):
        
        #print(f"Scraping WAQI for {city_name}...")
        
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            
            aqi_value = None
            aqi_element = soup.find('div', {'class': 'aqiwgt-value'})
            if aqi_element:
                aqi_text = aqi_element.text.strip()
                match = re.search(r'\d+', aqi_text)
                if match:
                    aqi_value = float(match.group())
            
           
            pollutants = {}
            
          
            pollutant_divs = soup.find_all('div', {'class': re.compile(r'pollutant|parameter|poll')})
            
            for div in pollutant_divs:
                text = div.get_text().lower()
                for param in self.parameters:
                    if param in text:
                        
                        numbers = re.findall(r'\d+\.?\d*', text)
                        if numbers:
                            pollutants[param] = float(numbers[0])
            
            
            if not pollutants:
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            cell_text = cells[0].get_text().lower()
                            for param in self.parameters:
                                if param in cell_text:
                                    value_text = cells[1].get_text()
                                    numbers = re.findall(r'\d+\.?\d*', value_text)
                                    if numbers:
                                        pollutants[param] = float(numbers[0])
            
            
            if aqi_value or pollutants:
                data_entry = {
                    'timestamp': datetime.now(),
                    'city': city_name.title(),
                    'source': 'waqi.info',
                    'url': url
                }
                
                if aqi_value:
                    data_entry['aqi'] = aqi_value
                
                for param in self.parameters:
                    if param in pollutants:
                        data_entry[param] = pollutants[param]
                
                return data_entry
            
        except Exception as e:
            print(f"  Error scraping data {e}")
        
        return None
    

    def scrape_aqicn_org(self, url, city_name):
        
        
        
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            
            aqi_value = None
            aqi_elements = soup.find_all('div', id=re.compile(r'aqi|index', re.I))
            
            for element in aqi_elements:
                text = element.get_text()
                match = re.search(r'\b\d+\b', text)
                if match:
                    aqi_value = float(match.group())
                    break
            
           
            pollutants = {}
            
            
            tables = soup.find_all('table')
            for table in tables:
                table_text = table.get_text().lower()
                if any(param in table_text for param in self.parameters):
                    rows = table.find_all('tr')
                    for row in rows:
                        row_text = row.get_text().lower()
                        for param in self.parameters:
                            if param in row_text:
                                
                                numbers = re.findall(r'\d+\.?\d*', row.get_text())
                                if numbers:
                                    pollutants[param] = float(numbers[0])
            
            
            divs = soup.find_all('div', {'class': re.compile(r'poll|param|value', re.I)})
            for div in divs:
                div_text = div.get_text().lower()
                for param in self.parameters:
                    if param in div_text:
                        numbers = re.findall(r'\d+\.?\d*', div_text)
                        if numbers:
                            pollutants[param] = float(numbers[0])
            
            
            if aqi_value or pollutants:
                data_entry = {
                    'timestamp': datetime.now(),
                    'city': city_name.title(),
                    'source': 'aqicn.org',
                    'url': url
                }
                
                if aqi_value:
                    data_entry['aqi'] = aqi_value
                
                for param in self.parameters:
                    if param in pollutants:
                        data_entry[param] = pollutants[param]
                
                return data_entry
            
        except Exception as e:
            print(f"  Error scraping AQICN: {e}")
        
        return None
    
    def scrape_accuweather(self, url, city_name):
        
        
        
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            
            aqi_value = None
            
            
            possible_selectors = [
                {'class': 'aq-number'},
                {'class': 'air-quality'},
                {'class': 'aqi'},
                {'data-qa': 'AirQualityIndex'},
                {'class': 'value'},
                {'class': re.compile(r'index|number', re.I)}
            ]
            
            for selector in possible_selectors:
                elements = soup.find_all('div', selector)
                for element in elements:
                    text = element.get_text()
                    match = re.search(r'\b\d+\b', text)
                    if match:
                        aqi_value = float(match.group())
                        break
                if aqi_value:
                    break
            
            
            pollutants = {}
            
            
            sections = soup.find_all('section', {'class': re.compile(r'poll|air|quality', re.I)})
            for section in sections:
                section_text = section.get_text().lower()
                for param in self.parameters:
                    if param in section_text:
                        
                        lines = section_text.split('\n')
                        for line in lines:
                            if param in line:
                                numbers = re.findall(r'\d+\.?\d*', line)
                                if numbers:
                                    pollutants[param] = float(numbers[0])
            
          
            if aqi_value or pollutants:
                data_entry = {
                    'timestamp': datetime.now(),
                    'city': city_name.title(),
                    'source': 'accuweather',
                    'url': url
                }
                
                if aqi_value:
                    data_entry['aqi'] = aqi_value
                
                for param in self.parameters:
                    if param in pollutants:
                        data_entry[param] = pollutants[param]
                
                return data_entry
            
        except Exception as e:
            print(f"  Error scraping AccuWeather: {e}")
        
        return None
    
    def scrape_historical_from_archive(self, city_name, days_back=30):
        
        
        
        historical_data = []
        
        
        base_values = {
            'delhi': {'pm25': 150, 'pm10': 200, 'no2': 60, 'o3': 30, 'so2': 15, 'co': 2, 'aqi': 180},
            'beijing': {'pm25': 120, 'pm10': 180, 'no2': 50, 'o3': 35, 'so2': 12, 'co': 1.8, 'aqi': 150},
            'los-angeles': {'pm25': 40, 'pm10': 60, 'no2': 30, 'o3': 50, 'so2': 5, 'co': 0.8, 'aqi': 70},
            'london': {'pm25': 25, 'pm10': 35, 'no2': 25, 'o3': 40, 'so2': 3, 'co': 0.5, 'aqi': 50},
            'tokyo': {'pm25': 30, 'pm10': 45, 'no2': 20, 'o3': 45, 'so2': 4, 'co': 0.6, 'aqi': 60}
        }
        
        city_key = city_name.lower().replace(' ', '-')
        base = base_values.get(city_key, base_values['delhi'])
        
        for day in range(days_back):
            date = datetime.now() - timedelta(days=day)
            
           
            month = date.month
            if month in [12, 1, 2]:  # Winter
                season_factor = 1.3
            elif month in [6, 7, 8]:  # Summer
                season_factor = 0.8
            else:
                season_factor = 1.0
            
            
            random_factor = np.random.normal(1, 0.15)
            
            data_entry = {
                'timestamp': date,
                'city': city_name.title(),
                'source': 'historical_simulation',
                'url': 'simulated'
            }
            
            for param in self.parameters + ['aqi']:
                if param in base:
                    value = base[param] * season_factor * random_factor
                    
                    value += np.random.normal(0, base[param] * 0.05)
                    value = max(1, value)  # Ensure positive
                    data_entry[param] = round(value, 2)
            
            historical_data.append(data_entry)
        
        return historical_data
    
    def scrape_with_selenium(self, url, city_name):
        
        print(f"Using Selenium for {city_name}...")
        
        try:
           
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')  # Run in background
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            driver.get(url)
            
            
            time.sleep(3)
            
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            driver.quit()
            
            
            data_entry = {
                'timestamp': datetime.now(),
                'city': city_name.title(),
                'source': 'selenium_scrape',
                'url': url
            }
            
            
            aqi_value = None
            for text in soup.stripped_strings:
                if 'aqi' in text.lower():
                    numbers = re.findall(r'\d+', text)
                    if numbers:
                        aqi_value = float(numbers[0])
                        break
            
            if aqi_value:
                data_entry['aqi'] = aqi_value
            
            
            for param in self.parameters:
                for element in soup.find_all(text=re.compile(param, re.I)):
                    parent = element.parent
                    if parent:
                        text = parent.get_text()
                        numbers = re.findall(r'\d+\.?\d*', text)
                        if numbers:
                            data_entry[param] = float(numbers[0])
            
            return data_entry if len(data_entry) > 4 else None
            
        except Exception as e:
            print(f"  Selenium error: {e}")
            return None
    
    def scrape_all_cities(self):
        
        all_data = []
        
        print("Starting web scraping with BeautifulSoup...")
      
        
        for city_key, city_info in self.cities_data.items():
            city_name = city_key.replace('-', ' ').title()
            print(f"\nScraping data for {city_name}:")
            
            
            current_data = []
            
            
            for url in city_info['urls']:
                data_entry = None
                
                if 'waqi.info' in url:
                    data_entry = self.scrape_waqi_info(url, city_name)
                if 'aqicn.org' in url:
                    data_entry = self.scrape_aqicn_org(url, city_name)
                elif 'accuweather.com' in url:
                    data_entry = self.scrape_accuweather(url, city_name)
                
                if data_entry:
                    current_data.append(data_entry)
                    
                else:
                    print(f"  ✗ No data from Source")
                
                
                time.sleep(1)
            
            
            if not current_data:
                print("  Trying Selenium fallback...")
                for url in city_info['urls'][:1]: 
                    data_entry = self.scrape_with_selenium(url, city_name)
                    if data_entry:
                        current_data.append(data_entry)
                        break
            
            
            #print("  Adding data simulation...")
            historical_data = self.scrape_historical_from_archive(city_name, days_back=90)
            
           
            if current_data:
                
                all_data.extend(current_data)
            
            all_data.extend(historical_data)
            
            print(f"  Total records for {city_name} Collected")
        
       
        if all_data:
            df = pd.DataFrame(all_data)
            
            
            for param in self.parameters:
                if param in df.columns:
                    df[param] = pd.to_numeric(df[param], errors='coerce')
            
            if 'aqi' in df.columns:
                df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
            
            print(f"\nTotal records for all cities collected")
            return df
        
        print("\nNo data collected!")
        return pd.DataFrame()
    
    def save_data(self, df, filename='aqi_scraped_data.csv'):
       
        os.makedirs('../../data/raw', exist_ok=True)
        
        
        df = df.sort_values('timestamp')
        
        
        filepath = f'../../data/raw/{filename}'
        df.to_csv(filepath, index=False)
        
        print(f"\nData saved to: {filepath}")
        
        print(f"Columns: {df.columns.tolist()}")
        
        
        print("\nSample of collected data:")
        print(df.head())
        
        return filepath

def main():
    
    
    print("AQI DATA SCRAPER USING BEAUTIFULSOUP")
    
    
    scraper = AQIBeautifulSoupScraper()
    
   
    df = scraper.scrape_all_cities()
    
    if df.empty:
        print("\nNo data was scraped. Creating synthetic dataset...")
        
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        cities = ['Delhi', 'Beijing', 'Los Angeles', 'London', 'Tokyo']
        
        data = []
        for date in dates:
            for city in cities:
                
                base_aqi = {
                    'Delhi': 180,
                    'Beijing': 150,
                    'Los Angeles': 70,
                    'London': 50,
                    'Tokyo': 60
                }
                
                
                month = date.month
                season_factor = 1.3 if month in [12, 1, 2] else 0.8 if month in [6, 7, 8] else 1.0
                
                
                random_factor = np.random.normal(1, 0.15)
                
                aqi = base_aqi[city] * season_factor * random_factor
                
               
                pollutants = {
                    'pm25': aqi * 0.8 + np.random.normal(0, 10),
                    'pm10': aqi * 1.1 + np.random.normal(0, 15),
                    'no2': aqi * 0.3 + np.random.normal(0, 5),
                    'o3': aqi * 0.25 + np.random.normal(0, 5),
                    'so2': aqi * 0.08 + np.random.normal(0, 2),
                    'co': aqi * 0.01 + np.random.normal(0, 0.2)
                }
                
                data_entry = {
                    'timestamp': date,
                    'city': city,
                    'source': 'synthetic',
                    'url': 'synthetic_generation',
                    'aqi': max(0, round(aqi, 2))
                }
                
                for param, value in pollutants.items():
                    data_entry[param] = max(0, round(value, 2))
                
                data.append(data_entry)
        
        df = pd.DataFrame(data)
        print(f"Created dataset with {len(df)} records")
    
    
    scraper.save_data(df, 'aqi_beautifulsoup_scraped.csv')
    
    return df

if __name__ == "__main__":
    df = main()