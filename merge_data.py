import pandas as pd
import requests
import time
from rapidfuzz import process, utils
import io  # for safety, though not needed now

# Scrapes end-of-season predictive power rankings from teamrankings.com
# Fills missing 'RK' column for years 2013–2024 except 2020
# Uses fuzzy matching due to name variations

# Load data
df = pd.read_csv('cbb2.csv')

# Rename the first column from Unnamed: 0 to TEAM
if 'Unnamed: 0' in df.columns:
    df.rename(columns={'Unnamed: 0': 'TEAM'}, inplace=True)

# Update the team_col variable to match the new name
team_col = 'TEAM'

df.rename(columns={'year': 'YEAR'}, inplace=True)
df[team_col] = df[team_col].astype(str).str.strip()

# List of years (excluding 2020)
years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def get_best_match(name, choices):
    result = process.extractOne(name, choices, processor=utils.default_process, score_cutoff=80)
    return result[0] if result and result[1] >= 80 else None

print("Fetching performance-based rankings...")

for year in years: 
    url = f"https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other-type?date={year}-04-01"
    
    try:
        r = requests.get(url, headers=header)
        if r.status_code != 200:
            print(f"Year {year}: HTTP {r.status_code} - skipping")
            continue
        
        tables = pd.read_html(io.StringIO(r.text), flavor='bs4')
        if not tables:
            print(f"Year {year}: No tables found")
            continue
        
        stats = tables[0]
        
        # Clean team names: remove (W-L) record
        stats['Team_Clean'] = stats['Team'].str.replace(r'\s*\(\d+-\d+\)', '', regex=True).str.strip()
        
        # List of cleaned names from web
        web_schools = stats['Team_Clean'].tolist()
        
        # Dict: cleaned name → rank (integer)
        rank_map = dict(zip(stats['Team_Clean'], stats['Rank']))
        
        # Quick diagnostic: print top 3 and a sample match
        print(f"\nYear {year} sample ranks:")
        print(rank_map.get('Louisville', 'Not found'), rank_map.get('Florida', 'Not found'))
        
        # Matching loop
        mask = df['YEAR'] == year
        matched_count = 0
        
        for idx, row in df[mask].iterrows():
            csv_name = row[team_col]
            match = get_best_match(csv_name, web_schools)
            if match:
                df.at[idx, 'RK'] = rank_map[match]  # already int
                matched_count += 1
        
        print(f"Year {year}: Successfully matched {matched_count} / {mask.sum()} teams")
        
        # If matched < expected (~350 or whatever your df has per year), print unmatched for debug
        if matched_count < 200:  # arbitrary threshold
            unmatched = df[mask & df['RK'].isna()][team_col].tolist()[:10]
            print(f"  Sample unmatched: {unmatched}")
        
        time.sleep(2)  # polite delay

    except Exception as e:
        print(f"Error on {year}: {e}")
        time.sleep(3)

# 4. Final Save
df.to_csv('cbb2_ranked.csv', index=False)
print("\nDone! Check 'cbb2_ranked.csv' for the new 'RK' column.")