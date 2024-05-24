from flask import Flask, request, jsonify
from threading import Thread
import random
import time
from faker import Faker
from datetime import datetime, timedelta, date
import os
import secrets

os.environ['TZ'] = 'UTC'  # Set timezone to UTC

app = Flask(__name__)

# Global variable to store the latest data
latest_data = None

# Initialize Faker
faker_instance = Faker()

# Define peak hours
PEAK_HOURS = [8, 9, 12, 13, 17, 18, 21, 22]

# User storage (in-memory for simplicity)
user_storage = {}
token_storage = {}

def random_timestamp():
    """
    Generate a random timestamp for the current day or a given day.
    """
    eleven_days_ago = date.today() - timedelta(days=11)
    random_date = eleven_days_ago + timedelta(days=random.randint(0, 11))
    hour = random.choice(PEAK_HOURS)
    minute, second = random.randint(0, 59), random.randint(0, 59)
    return f"{random_date.strftime('%d/%m/%Y')} {hour:02d}:{minute:02d}:{second:02d}"

# Data Dictionary
DATA_DICT = {
    'request': ['GET'],
    'resource': [
        '/register', '/register/confirmation', '/login', '/logout', '/user/profile', '/user/settings', 
        '/user/favorites', '/user/watch-later', '/events/aquatics/details', '/events/archery/details',
        '/events/athletics/details', '/events/badminton/details', '/events/basketball/details', 
        '/events/boxing/details', '/events/canoe/details', '/events/cycling/details', '/events/equestrian/details', 
        '/events/fencing/details', '/events/football/details', '/events/golf/details', '/events/gymnastics/details', 
        '/events/handball/details', '/events/hockey/details', '/events/judo/details', '/events/modern_pentathlon/details', 
        '/events/rowing/details', '/events/rugby/details', '/events/sailing/details', '/events/shooting/details', 
        '/events/skateboarding/details', '/events/sport_climbing/details', '/events/surfing/details', 
        '/events/table_tennis/details', '/events/taekwondo/details', '/events/tennis/details', '/events/triathlon/details', 
        '/events/volleyball/details', '/events/weightlifting/details', '/events/wrestling/details', '/schedule', 
        '/medal-tally', '/news/latest', '/video/highlights', '/contact', '/about-us'
    ],
    'statuscode': ['200', '304'],
    'user_agent': [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4380.0 Safari/537.36 Edg/89.0.759.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36 OPR/73.0.3856.329',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A'
    ],
    'ip_address': [
        '87.201.198.48', '31.218.216.241', '129.122.161.84', '201.231.47.242', '186.152.163.216', '163.10.238.207',
        '190.111.57.203', '186.148.113.126', '181.22.46.113', '80.110.152.91', '195.64.20.149', '220.236.74.119',
        '124.168.151.142', '139.86.39.1', '134.211.11.69', '61.247.177.253', '87.65.236.224', '84.197.7.123',
        '57.219.211.196', '78.20.197.245', '178.145.218.214', '187.4.39.112', '177.52.209.111', '181.215.28.143',
        '177.90.131.150', '152.237.17.16', '50.23.186.142', '129.128.183.87', '74.82.84.108', '100.26.205.115',
        '92.62.182.106', '57.9.13.19', '192.111.48.232', '57.15.250.184', '200.83.202.252', '43.16.198.54',
        '115.152.69.52', '140.237.166.240', '106.229.97.88', '101.207.183.97', '112.89.241.213', '191.92.181.26',
        '152.203.151.117', '200.116.69.87', '190.70.236.6', '201.207.185.198', '201.193.238.121', '160.217.161.141',
        '82.142.68.215', '160.70.151.179', '195.52.94.4', '159.25.59.234', '194.124.200.244', '2.129.46.164',
        '37.96.42.51', '152.73.88.50', '38.44.63.61', '181.78.196.122', '45.243.27.243', '196.139.234.217',
        '154.185.38.118', '41.65.71.113', '156.182.169.15', '156.216.169.146', '217.127.152.184', '79.145.190.111',
        '88.24.185.179', '206.204.154.247', '145.247.88.139', '151.105.134.154', '128.214.250.217', '80.220.114.120',
        '193.167.58.194', '62.100.138.125', '78.225.4.137', '86.202.93.85', '86.233.206.93', '164.131.79.144',
        '80.2.6.156', '193.128.150.31', '176.250.178.38', '48.221.106.59', '94.43.203.167', '197.191.151.13',
        '46.177.76.113', '168.123.199.1', '4.192.106.207', '154.23.122.58', '168.106.117.44', '42.3.241.180',
        '154.223.128.15', '154.213.248.71', '84.225.206.245', '182.11.216.165', '139.194.237.82', '36.76.17.63',
        '94.199.224.104', '87.232.240.98', '191.237.215.253', '163.77.164.182', '77.139.133.190', '77.127.219.254',
        '154.94.217.83', '103.231.43.195', '27.60.20.7', '121.246.191.3', '139.38.179.59', '165.8.205.201',
        '41.149.142.211', '10.83.251.121', '172.31.128.200', '10.103.161.56'
    ]
}

def generate_user_data(num_logs=6000):
    """
    Generate user data for the current day and the past 11 days.
    """
    user_data_list = [
        f"{random.choice(DATA_DICT['ip_address'])} - - [{random_timestamp()}] \"{random.choice(DATA_DICT['request'])} {random.choice(DATA_DICT['resource'])} HTTP/1.0\" {random.choice(DATA_DICT['statuscode'])} {random.randint(100, 5000)} \"-\" \"{faker_instance.user_agent()}\" {random.randint(1, 36)}"
        for _ in range(num_logs * 11)
    ]
    return user_data_list

def update_data_in_background():
    global latest_data
    while True:
        latest_data = generate_user_data()
        time.sleep(3)  # Update data every 3 seconds (adjust as needed)

def generate_token():
    return secrets.token_urlsafe(32)

# Create and start the background thread
data_thread = Thread(target=update_data_in_background)
data_thread.daemon = True
data_thread.start()

@app.route('/api/register', methods=['POST'])
def register_user():
    user_details = request.json
    username = user_details.get('username')
    if not username:
        return jsonify({"error": "Username is required"}), 400
    
    if username in user_storage:
        return jsonify({"error": "User already exists"}), 409
    
    token = generate_token()
    user_storage[username] = {
        "username": username,
        "token": token
    }
    token_storage[token] = username
    return jsonify({"username": username, "token": token})

@app.route('/api/olympicdata', methods=['GET'])
def get_olympic_data():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Missing or invalid authorization token"}), 401

    token = auth_header.split()[1]
    if token not in token_storage:
        return jsonify({"error": "Invalid API token"}), 401

    return jsonify(latest_data)

@app.route('/api/token', methods=['GET'])
def generate_api_token():
    token = generate_token()
    return jsonify({"token": token})

@app.route('/')
def FunOlympics():
    return "FunOlympics Mock API"

if __name__ == '__main__':
    app.run(debug=True)
