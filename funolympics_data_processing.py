import requests
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def get_olympic_data(api_token: str, url: str, limit: int = 12000) -> list:
    headers = {'Authorization': f'Bearer {api_token}'}
    response = requests.get(url, headers=headers, params={'limit': limit})
    if response.status_code == 200:
        return response.json()[:limit]  # Limit the number of rows
    else:
        logging.error(f"Error fetching data: {response.status_code}")
        return []

def log_data(data: list, log_file_path: str) -> None:
    with open(log_file_path, "a") as f:
        for log_entry in data:
            f.write(f"{log_entry}\n")

def parse_log_data(data: list) -> pd.DataFrame:
    def parse_log_line(line: str) -> dict:
        parts = line.split()
        date_timestamp = parts[3][1:]
        timestamp = parts[4][1:-1]
        IP = parts[0]
        request = parts[5].split("/")
        request_method = request[0][1:]
        resource = parts[6]
        status_code = int(parts[8])
        response_size = int(parts[9])
        user_agent = ' '.join(parts[11:-1])  # Combine the user agent parts
        elapsed_time = parts[-1]
        
        return {
            "timestamp": f"{date_timestamp} {timestamp}",
            "IP": IP,
            "request_method": request_method,
            "resource": resource,
            "status_code": status_code,
            "response_size": response_size,
            "user_agent": user_agent,
            "elapsed_time": elapsed_time
        }
    
    new_data = [parse_log_line(line) for line in data]
    return pd.DataFrame(new_data)

def get_event(resource):
    if resource.startswith('/events/'):
        event_type = resource.split('/')[2]
        return f"{event_type} event"
    return 'Other'

country_code_map = {
    "ID": "Indonesia", "PL": "Poland", "ZA": "South Africa", "ZZ": "Invalid Territory",
    "CA": "Canada", "IN": "India", "US": "United States of America", "LT": "Lithuania",
    "TR": "Turkey", "SG": "Singapore", "BR": "Brazil", "FI": "Finland", "AR": "Argentina",
    "BD": "Bangladesh", "DK": "Denmark", "KE": "Kenya", "MX": "Mexico", "SD": "Sudan",
    "MA": "Morocco", "DE": "Germany", "BE": "Belgium", "RU": "Russia", "NO": "Norway",
    "IT": "Italy", "UA": "Ukraine", "AU": "Australia", "PH": "Philippines", "IE": "Ireland",
    "PK": "Pakistan", "TN": "Tunisia", "VN": "Vietnam", "CR": "Costa Rica", "CH": "Switzerland",
    "RO": "Romania", "GB": "Great Britain", "DO": "Dominican Republic", "CN": "China",
    "NL": "Netherlands", "AT": "Austria", "CL": "Chile", "MU": "Mauritius", "GE": "Georgia",
    "ES": "Spain", "RS": "Serbia", "SE": "Sweden", "FR": "France", "JP": "Japan",
    "PT": "Portugal", "HK": "Hong Kong", "NZ": "New Zealand", "CO": "Colombia", "VE": "Venezuela",
    "GR": "Greece", "AO": "Angola", "RE": "Réunion", "PE": "Peru", "IR": "Iran", "KW": "Kuwait",
    "AE": "United Arab Emirates", "CZ": "Czech Republic", "GU": "Guam", "KR": "South Korea",
    "TW": "Taiwan", "IL": "Israel", "EG": "Egypt", "EC": "Ecuador", "GH": "Ghana", "NA": "Namibia",
    "SK": "Slovakia", "TH": "Thailand", "NG": "Nigeria", "HU": "Hungary"
}

def get_full_country_name(code):
    return country_code_map.get(code, 'Unknown')

def get_continent_from_country(country_code):
    # Map country codes to continents
    continent_map = {
    'US': 'North America', 'CA': 'North America', 'MX': 'North America', 'CR': 'North America', 'DO': 'North America', 'GU': 'North America',
    'BR': 'South America', 'AR': 'South America', 'CO': 'South America', 'CL': 'South America', 'VE': 'South America', 'PE': 'South America',
    'GB': 'Europe', 'FR': 'Europe', 'DE': 'Europe', 'PL': 'Europe', 'LT': 'Europe', 'FI': 'Europe', 'DK': 'Europe', 'BE': 'Europe', 'NO': 'Europe',
    'IT': 'Europe', 'UA': 'Europe', 'RO': 'Europe', 'NL': 'Europe', 'AT': 'Europe', 'ES': 'Europe', 'RS': 'Europe', 'SE': 'Europe', 'PT': 'Europe',
    'CZ': 'Europe', 'IE': 'Europe', 'GR': 'Europe', 'SK': 'Europe', 'HU': 'Europe', 'RU': 'Europe',
    'CN': 'Asia', 'JP': 'Asia', 'IN': 'Asia', 'ID': 'Asia', 'TR': 'Asia', 'SG': 'Asia', 'BD': 'Asia', 'PH': 'Asia', 'VN': 'Asia', 'GE': 'Asia', 
    'IR': 'Asia', 'KW': 'Asia', 'AE': 'Asia', 'KR': 'Asia', 'TW': 'Asia', 'IL': 'Asia', 'TH': 'Asia',
    'AU': 'Oceania', 'NZ': 'Oceania', 'NA': 'Oceania',
    'ZA': 'Africa', 'NG': 'Africa', 'EG': 'Africa', 'KE': 'Africa', 'SD': 'Africa', 'MA': 'Africa', 'TN': 'Africa', 'AO': 'Africa', 'GH': 'Africa', 'RE': 'Africa',
    'ZZ': 'Invalid Territory'
    }
    return continent_map.get(country_code, 'Other')

def group_resources(resource):
    if resource.startswith('/user'):
        return 'User Management'
    elif resource.startswith('/events'):
        return 'Event Details'
    elif resource in ['/schedule', '/medal-tally', '/news/latest', '/video/highlights']:
        return 'Informational Pages'
    return 'Other'

def group_sports(resource):
    if resource.startswith('/events'):
        sport = resource.split('/')[2]
        if sport in ['tennis', 'badminton', 'table_tennis']:
            return 'Racquet Sports'
        elif sport in ['boxing', 'judo', 'wrestling']:
            return 'Combat Sports'
        elif sport in ['football', 'basketball', 'volleyball']:
            return 'Ball Sports'
        elif sport in ['aquatics', 'sailing']:
            return 'Water Sports'
        elif sport in ['archery', 'shooting']:
            return 'Precision Sports'
        elif sport in ['athletics', 'cycling', 'rowing']:
            return 'Endurance Sports'
        elif sport in ['weightlifting', 'wrestling']:
            return 'Strength Sports'
        elif sport in ['modern_pentathlon']:
            return 'Strategy Sports'
        elif sport in ['sport_climbing', 'surfing', 'skateboarding']:
            return 'Adventure Sports'
        return 'Other Sports'
    return 'Non-sport'

def extract_device_os_browser(user_agent):
    device = 'Unknown'
    os = 'Unknown'
    browser = 'Unknown'

    if 'Android' in user_agent:
        device = 'Mobile'
        os = 'Android'
    elif 'iPhone' in user_agent or 'iPod' in user_agent:
        device = 'Mobile'
        os = 'iOS'
    elif 'Windows' in user_agent:
        os = 'Windows'
        if 'Phone' in user_agent:
            device = 'Mobile'
        else:
            device = 'Desktop'
    elif 'Linux' in user_agent:
        os = 'Linux'
        if 'X11' in user_agent:
            device = 'Desktop'
        else:
            device = 'Mobile'
    elif 'Mac' in user_agent:
        os = 'MacOS'
        if 'PPC' in user_agent:
            device = 'Desktop'
        else:
            device = 'Mobile'

    if 'MSIE' in user_agent:
        browser = 'Internet Explorer'
    elif 'Opera' in user_agent:
        browser = 'Opera'
    elif 'Chrome' in user_agent:
        browser = 'Chrome'
    elif 'Safari' in user_agent:
        browser = 'Safari'
    elif 'Firefox' in user_agent:
        browser = 'Firefox'

    return device, os, browser

def process_data(api_token: str, url: str) -> pd.DataFrame:
    data = get_olympic_data(api_token, url, limit=12000)
    parsed_data = parse_log_data(data)
    df = pd.DataFrame(parsed_data)
    IP_INFO = pd.read_csv('ipinfoData.csv')

    # Merge the DataFrames based on the 'IP' column
    merged_df = pd.merge(df, IP_INFO, how='left', on='IP')

    # Fill missing values in the merged DataFrame
    merged_df.fillna({
        'Country': 'Unknown',
        'Region': 'Unknown',
        'City': 'Unknown',
        'Latitude': 0.0,
        'Longitude': 0.0
    }, inplace=True)

    # Convert the 'timestamp' column to datetime
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], format='%d/%m/%Y %H:%M:%S')
    merged_df.sort_values(by=['IP', 'timestamp'], inplace=True)

    # Define a session timeout threshold (e.g., 30 minutes)
    SESSION_TIMEOUT = 30 * 60 * 1000  # 30 minutes in milliseconds

    # Calculate session duration
    merged_df['previous_timestamp'] = merged_df.groupby('IP')['timestamp'].shift(1)
    merged_df['session_duration'] = (merged_df['timestamp'] - merged_df['previous_timestamp']).dt.total_seconds() * 1000
    merged_df['session_duration'] = merged_df['session_duration'].apply(lambda x: x if x <= SESSION_TIMEOUT else 0)
    merged_df['session_duration'].fillna(0, inplace=True)

    # Apply the function to the user_agent column
    merged_df['Device'], merged_df['Operating_System'], merged_df['Browser'] = zip(*merged_df['user_agent'].apply(extract_device_os_browser))

    # Apply the function to create the Event column
    merged_df['Event'] = merged_df['resource'].apply(get_event)

    # Create a new column named 'Full Country Name'
    merged_df['Full Country Name'] = merged_df['Country'].apply(get_full_country_name)
    merged_df['Continent'] = merged_df['Country'].apply(get_continent_from_country)

    # Group resources and sports
    merged_df['resource_group'] = merged_df['resource'].apply(group_resources)
    merged_df['sports_group'] = merged_df['resource'].apply(group_sports)

    return merged_df