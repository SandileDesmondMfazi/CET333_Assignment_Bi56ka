import streamlit as st
import pandas as pd
import funolympics_data_processing  # Import the data processing script
import numpy as np
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta, date
import time
import plotly.graph_objects as go
import requests
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="FunOlympics Games Analytics Dashboard",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Define initial DataFrame and history
df = pd.DataFrame()
history = []

# Function to process data
def process_data(api_token, url):
    new_df = funolympics_data_processing.process_data(api_token, url)
    return pd.concat([df, new_df.loc[:np.random.choice(range(4500, 5000))]], ignore_index=True)

# Function to filter data
def filter_data(df):
    return df[(df['Country'] != 'ZZ') & (df['Full Country Name'] != 'Other')]

def events_filter(df):
    return df[df['Event'] != 'Other']

# Import the requests library
import requests

# Function to check if the API token is valid
def check_token(token, url):
    """
    Function to check if the API token is valid by sending a request to the server.
    """
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return True  # Token is valid
    else:
        return False  # Token is invalid

# Function to show login form
def show_login_form():
    # Display the login form
    with st.form("login_form"):
        url = st.text_input("Enter API URL")
        token = st.text_input("Enter API Token", type="password")
        submit = st.form_submit_button("Submit")

        # When the form is submitted
        if submit:
            # Check if the token is valid
            if check_token(token, url):
                st.session_state["API_URL"] = url
                st.session_state["API_TOKEN"] = token
                st.session_state["logged_in"] = True
                st.success("Login successful! Proceeding to dashboard...")
                # Proceed to the dashboard or any other action
            else:
                st.error("Invalid API token. Please try again.")

# Check if credentials are stored in session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Show login form if not logged in
if not st.session_state["logged_in"]:
    show_login_form()
else:
    # Navigation bar
    selected = option_menu(
        menu_title="",
        options=["Dashboard", "Data View", "Predict Views"],
        icons=["house", "table", "Chart"],
        menu_icon="",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#222222"},  # Dark background
            "icon": {"color": "white", "font-size": "15px"},  # Light font color
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px 1px", "padding": "5px", "color": "lightgrey"},  # Adjusted padding, margin, and font color
            "nav-link-selected": {"background-color": "#696868", "color": "black"}  # Changed background color and text color for selected link
        }
    )

    API_TOKEN = st.session_state["API_TOKEN"]
    URL = st.session_state["API_URL"]

    if selected == "Dashboard":
        # Process new data (initial data with past 11 days)
        df = filter_data(process_data(API_TOKEN, URL))
        start_date = pd.to_datetime(datetime.now() - timedelta(days=10))  # Simulate past 11 days of data
        end_date = start_date + timedelta(days=10)  # Simulate next 10 days
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

        # Sidebar
        with st.sidebar:
            st.title('🥇FunOlympics Games Dashboard')

            # Filter by continent
            continent_options = sorted(df['Continent'].unique())
            continent_options.insert(0, "All")
            selected_continent = st.sidebar.selectbox('Continent', continent_options, index=0, key='continent_key')

            # Filter by Olympic event type
            event_type_options = sorted(df['Event_Type'].unique())  # Assuming 'Event_Type' is the column for Olympic event type
            event_type_options.insert(0, "All")
            selected_event_type = st.sidebar.selectbox('Olympic Event Type', event_type_options, index=0, key='event_type_key')

            device_options = sorted(df['Device'].unique())
            device_options.insert(0, "All")
            selected_device = st.sidebar.selectbox('Device', device_options, index=0, key='device_key')

        def calculate_kpis(df):
            # Calculate total visits and average duration
            total_visits = len(df)
            avg_duration = df['session_duration'].mean()

            # Calculate most visited resource
            most_visited_resource = df['Event'].value_counts().idxmax() if not df['Event'].empty else 'N/A'

            # Calculate top country
            top_country = df['Full Country Name'].value_counts().idxmax() if not df['Full Country Name'].empty else 'N/A'

            # Calculate live visits
            current_time = datetime.now()
            start_time = current_time - timedelta(days=1)  # Consider the last minute as "live"
            live_visits = len(df[(df['timestamp'] >= start_time) & (df['timestamp'] < current_time)])

            return {
                'total_visits': total_visits,
                'avg_duration': avg_duration,
                'most_visited_resource': most_visited_resource,
                'top_country': top_country,
                'live_visits': live_visits  # New KPI
            }

        def display_kpi(kpi_name, value, delta, placeholder):
            placeholder.metric(label=kpi_name, value=value, delta=delta)

        def display_kpis(kpis, prev_kpis, placeholders):
            display_kpi("Total Visits", kpis['total_visits'], kpis['total_visits'] - prev_kpis.get('total_visits', 0), placeholders[0])
            display_kpi("Avg. Time Spent", convert_millis_to_mm_ss(kpis['avg_duration']), convert_millis_to_mm_ss(kpis['avg_duration'] - prev_kpis.get('avg_duration', 0)), placeholders[1])
            display_kpi("Most Visited Event", kpis['most_visited_resource'], "", placeholders[2])
            display_kpi("Top Country", kpis['top_country'], "", placeholders[3])
            display_kpi("Live Visits", kpis['live_visits'], "", placeholders[4])

        def convert_millis_to_mm_ss(millis):
            millis = int(millis)
            minutes = int(millis / (1000 * 60))
            seconds = int((millis % (1000 * 60)) / 1000)
            return f"{minutes:02d}:{seconds:02d}"

        def create_device_chart(df):
            device_counts = df.groupby('Device')['Device'].count().reset_index(name='Count')
            device_counts = device_counts.sort_values(by='Count', ascending=False)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=device_counts['Device'], y=device_counts['Count'], name="Device Count"))
            fig.update_layout(title="Top Devices", xaxis_title="Device", yaxis_title="Count")
            return fig

        def create_resource_group_chart(df):
            resource_group_counts = df['resource_group'].value_counts()

            fig = go.Figure()
            fig.add_trace(go.Pie(labels=resource_group_counts.index, values=resource_group_counts.values, textinfo='percent+label', textposition='outside', insidetextorientation='radial', hole=.75, name="Visits by Resource Group"))
            fig.update_layout(title="Distribution of Visits by Resource Group",showlegend=False)
            return fig

        def create_resource_chart(df):
            df1 = events_filter(df)
            resource_counts = df1['Event'].value_counts().nlargest(10)  # Get the top 10 events
            fig = go.Figure(data=[go.Bar(x=resource_counts.values, y=resource_counts.index, orientation='h')])
            fig.update_layout(
                title="Top 10 Events by Visits",
                xaxis_title="Visit Count",
                yaxis_title="Event",
                yaxis=dict(autorange="reversed")  # Order the y-axis by values in descending order
            )
            return fig


        def create_sport_group_chart(df):
            sport_df = df[df["sports_group"] != "Non-sport"]
            sport_group_counts = sport_df['sports_group'].value_counts()

            fig = go.Figure()
            fig.add_trace(go.Pie(labels=sport_group_counts.index, values=sport_group_counts.values, textinfo='percent+label', textposition='outside', insidetextorientation='radial', hole=.75, name="Visits by Sport Groups"))
            fig.update_layout(title="Distribution of Visits by Sport Groups", showlegend=False)
            return fig

        def create_continent_chart(df):
            continent_counts = df['Continent'].value_counts().nlargest(10)  # Get the top 10 continents
            fig = go.Figure(data=[go.Bar(x=continent_counts.values, y=continent_counts.index, orientation='h')])
            fig.update_layout(
                title="Top Continents by Visits",
                xaxis_title="Visit Count",
                yaxis_title="Continent",
                yaxis=dict(autorange="reversed")  # Order the y-axis by values in descending order
            )
            return fig



        def create_line_chart(df, static_df):
            static_df['date'] = static_df['timestamp'].dt.date
            static_df_daily = static_df.groupby('date').size().reset_index(name='visits')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=static_df_daily['date'], y=static_df_daily['visits'], name='Visits', connectgaps=True))

            current_day = pd.to_datetime(date.today())
            current_day_df = df[df['timestamp'].dt.date == current_day.date()]
            if not current_day_df.empty:
                current_day_visits = len(current_day_df)
                fig.add_trace(go.Scatter(x=[current_day], y=[current_day_visits], mode='markers', name='Current Day', connectgaps=True))

            fig.update_layout(
                title_text="Visits Over Time",
                xaxis_title="Date",
                yaxis_title="Visits"
            )
            return fig
        
        # Function to create a vertical bar chart for top 15 countries by visit count
        def create_country_visit_chart(df):
            top_countries = df['Full Country Name'].value_counts().nlargest(15)  # Get the top 15 countries by visit count
            fig = go.Figure(data=[go.Bar(x=top_countries.index, y=top_countries.values)])
            fig.update_layout(
                title="Top 15 Countries by Visit Count",
                xaxis_title="Country",
                yaxis_title="Visit Count"
            )
            return fig

        # Initialize session state
        if "session_state" not in st.session_state:
            st.session_state["session_state"] = {"df": df, "kpis": {}}

        # Create placeholders for the KPIs
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        total_visits_placeholder = kpi1.empty()
        avg_duration_placeholder = kpi2.empty()
        most_visited_resource_placeholder = kpi3.empty()
        top_country_placeholder = kpi4.empty()
        live_visits_placeholder = kpi5.empty()

        # Placeholder for the clock
        now_placeholder = st.empty()

        # Create placeholders for the charts
        col1, col2 = st.columns(2)
        sport_group_placeholder = col2.empty()
        resource_group_placeholder = col1.empty()
        col3, col4 = st.columns(2)
        resource_placeholder = col3.empty()
        continent_placeholder = col4.empty()
        country_visit_chart_placeholder = st.empty()
        line_chart_placeholder = st.empty()

        
        def update_clock():
            now = datetime.now().strftime("%H:%M:%S")
            now_placeholder.markdown(f"**Live Clock:** {now}")

        # Simulate new data for the current day (12th day)
        def simulate_new_data(df):
            current_day = pd.to_datetime(date.today())
            new_visits = np.random.randint(1, 5)
            new_data = {
                'timestamp': [current_day] * new_visits,
                'session_duration': np.random.randint(1000, 300000, new_visits),
                'Device': np.random.choice(df['Device'].unique(), new_visits),
                'Country': np.random.choice(df['Country'].unique(), new_visits),
                'Event': np.random.choice(df['Event'].unique(), new_visits),
                'Full Country Name': np.random.choice(df['Full Country Name'].unique(), new_visits),
                'sports_group': np.random.choice(df['sports_group'].unique(), new_visits),
                'resource_group': np.random.choice(df['resource_group'].unique(), new_visits)
            }
            new_df = pd.DataFrame(new_data)
            return pd.concat([df, new_df], ignore_index=True)

        # Initialize static data DataFrame outside the loop (assuming start_date is set to May 6th)
        static_df = filter_data(df)[(df['timestamp'] >= start_date) & (df['timestamp'] <= (start_date + timedelta(days=9)))]  # Past 10 days
        static_df['visits'] = np.random.randint(500, 2000, size=len(static_df))

        while True:
            try:
                # Update clock
                update_clock()

                # Process new data
                df = filter_data(process_data(API_TOKEN, URL))
                df = simulate_new_data(df)
                st.session_state["session_state"]["df"] = df

                df_filtered = df.copy()
                if selected_continent != "All":
                    df_filtered = df_filtered[df_filtered['Continent'] == selected_continent]
                if selected_event_type != "All":
                    df_filtered = df_filtered[df_filtered['Event_Type'] == selected_event_type]
                if selected_device != "All":
                    df_filtered = df_filtered[df_filtered['Device'] == selected_device]

                # Calculate KPIs
                kpis = calculate_kpis(events_filter(filter_data(df_filtered)))
                prev_kpis = st.session_state.get("session_state", {}).get("kpis", {})
                display_kpis(kpis, prev_kpis, [total_visits_placeholder, avg_duration_placeholder, most_visited_resource_placeholder, top_country_placeholder, live_visits_placeholder])

                # Update session state
                st.session_state["session_state"]["kpis"] = kpis

                # Update charts
                sport_group_placeholder.plotly_chart(create_sport_group_chart(df_filtered), use_container_width=True)
                resource_group_placeholder.plotly_chart(create_resource_group_chart(df_filtered), use_container_width=True)
                resource_placeholder.plotly_chart(create_resource_chart(filter_data(df_filtered)), use_container_width=True)
                line_chart_placeholder.plotly_chart(create_line_chart(filter_data(df_filtered), filter_data(static_df)), use_container_width=True)
                continent_placeholder.plotly_chart(create_continent_chart(df_filtered), use_container_width=True)
                country_visit_chart_placeholder.plotly_chart(create_country_visit_chart(df_filtered), use_container_width=True)

                # Pause for 3 seconds
                time.sleep(3)

            except Exception as e:
                st.error(f"Error: {e}")
                time.sleep(3)
    elif selected == "Data View":
        #st.title("Data View")

        # Data View content
        # Process new data (initial data with past 11 days)
        df = filter_data(process_data(API_TOKEN, URL))

        # Sidebar
        with st.sidebar:
            st.title('🥇FunOlympics Games Dashboard')

            # Filter by continent
            continent_options = sorted(df['Continent'].unique())
            continent_options.insert(0, "All")
            selected_continent = st.sidebar.selectbox('Continent', continent_options, index=0, key='continent_key')

            # Filter by Olympic event type
            event_type_options = sorted(df['Event_Type'].unique())  # Assuming 'Event_Type' is the column for Olympic event type
            event_type_options.insert(0, "All")
            selected_event_type = st.sidebar.selectbox('Olympic Event Type', event_type_options, index=0, key='event_type_key')

            device_options = sorted(df['Device'].unique())
            device_options.insert(0, "All")
            selected_device = st.sidebar.selectbox('Device', device_options, index=0, key='device_key')

        # Apply filters
        df_filtered = df.copy()
        if selected_continent != "All":
            df_filtered = df_filtered[df_filtered['Continent'] == selected_continent]
        if selected_event_type != "All":
            df_filtered = df_filtered[df_filtered['Event_Type'] == selected_event_type]
        if selected_device != "All":
            df_filtered = df_filtered[df_filtered['Device'] == selected_device]

        current_date = datetime.now().strftime("%Y%m%d")
        # Display the button for downloading the data
        csv = df.to_csv(index=False).encode("utf-8")

        # Offer the CSV file for download
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"Raw_Data{current_date}.csv",
            mime="text/csv",
        )
        st.write("Raw Data View")
        st.dataframe(df_filtered)

    elif selected == "Predict Views":
        # Process new data (initial data with past 11 days)
        df = filter_data(process_data(API_TOKEN, URL))

        # Sidebar
        with st.sidebar:
            st.title('🥇FunOlympics Games Dashboard')

            # Filter by continent
            continent_options = sorted(df['Continent'].unique())
            continent_options.insert(0, "All")
            selected_continent = st.sidebar.selectbox('Continent', continent_options, index=0, key='continent_key')

            # Filter by Olympic event type
            event_type_options = sorted(df['Event_Type'].unique())  # Assuming 'Event_Type' is the column for Olympic event type
            event_type_options.insert(0, "All")
            selected_event_type = st.sidebar.selectbox('Olympic Event Type', event_type_options, index=0, key='event_type_key')

            device_options = sorted(df['Device'].unique())
            device_options.insert(0, "All")
            selected_device = st.sidebar.selectbox('Device', device_options, index=0, key='device_key')

        # Apply filters
        df_filtered = df.copy()
        if selected_continent != "All":
            df_filtered = df_filtered[df_filtered['Continent'] == selected_continent]
        if selected_event_type != "All":
            df_filtered = df_filtered[df_filtered['Event_Type'] == selected_event_type]
        if selected_device != "All":
            df_filtered = df_filtered[df_filtered['Device'] == selected_device]

        # Data processing function
        def process_model_data(data):
            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            # Create a date column
            data['date'] = data['timestamp'].dt.date
            # Aggregate data by counting the number of requests
            aggregated_data = data.groupby(['date', 'Continent', 'Event_Type', 'Event']).size().reset_index(name='views')
            # Feature Engineering
            aggregated_data['day_of_week'] = pd.to_datetime(aggregated_data['date']).dt.dayofweek
            # Label Encoding
            le_continent = LabelEncoder()
            le_event_type = LabelEncoder()
            le_event = LabelEncoder()
            aggregated_data['continent_encoded'] = le_continent.fit_transform(aggregated_data['Continent'])
            aggregated_data['event_type_encoded'] = le_event_type.fit_transform(aggregated_data['Event_Type'])
            aggregated_data['event_encoded'] = le_event.fit_transform(aggregated_data['Event'])
            return aggregated_data, le_continent, le_event_type, le_event


        # Process data
        processed_data, le_continent, le_event_type, le_event = process_model_data(df)

        def train_model(data):
            features = ['day_of_week', 'continent_encoded', 'event_type_encoded', 'event_encoded']
            X = data[features]
            y = data['views']
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Model Training with Linear Regression
            model = LinearRegression()
            model.fit(X_train, y_train)
            # Model Evaluation
            score = model.score(X_test, y_test)
            st.write(f"Model R^2 score: {score:.2f}")
            return model, X_test, y_test

        # Train the model
        best_model, X_test, y_test = train_model(processed_data)


        # Predict function
        def predict_views(date, continent, event_type, event):
            # Prepare the input for prediction
            day_of_week = date.weekday()
            month = date.month
            day = date.day
            continent_encoded = le_continent.transform([continent])[0]
            event_type_encoded = le_event_type.transform([event_type])[0]
            event_encoded = le_event.transform([event])[0]
            input_features = [[day_of_week, month, day, continent_encoded, event_type_encoded, event_encoded]]
            # Prediction
            predicted_views = best_model.predict(input_features)[0]
            return predicted_views

        # Streamlit App
        st.title("Event View Prediction")
        date = st.date_input("Select Date")
        continent = st.selectbox("Continent", processed_data['Continent'].unique())
        event_type = st.selectbox("Event Type", processed_data['Event_Type'].unique())
        event = st.selectbox("Event", processed_data['Event'].unique())

        if st.button("Predict Views"):
            predicted_views = predict_views(date, continent, event_type, event)
            st.write(f"Predicted Views: {predicted_views:.0f}")
