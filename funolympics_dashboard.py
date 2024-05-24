import streamlit as st
import pandas as pd
import funolympics_data_processing  # Import the data processing script
import numpy as np
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta, date
import time
import plotly.graph_objects as go

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
    if "status_updated" not in st.session_state:
        st.session_state["status_updated"] = True
    new_df = funolympics_data_processing.process_data(api_token, url)
    return pd.concat([df, new_df.loc[:np.random.choice(range(4500, 5000))]], ignore_index=True)

# Function to filter data
def filter_data(df):
    return df[df['Country'] != 'ZZ']

def events_filter(df):
    return df[df['Event'] != 'Other']


# Function to show login form
def show_login_form():
    with st.form("login_form"):
        url = st.text_input("Enter API URL")
        token = st.text_input("Enter API Token", type="password")
        submit = st.form_submit_button("Submit")

        if submit:
            st.session_state["API_URL"] = url
            st.session_state["API_TOKEN"] = token
            st.session_state["logged_in"] = True
            st.success("Credentials submitted successfully")

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
            st.title('FunOlympics Games Dashboard')

            sport_category_options = sorted(df['sports_group'].unique())
            sport_category_options.insert(0, "All")
            selected_sport_category = st.sidebar.selectbox('Sport Category', sport_category_options, index=0, key='sport_category_key')

            device_options = sorted(df['Device'].unique())
            device_options.insert(0, "All")
            selected_device = st.sidebar.selectbox('Device', device_options, index=0, key='device_key')

            country_options = sorted(df['Full Country Name'].unique())
            country_options.insert(0, "All")
            selected_country = st.sidebar.selectbox('Country', country_options, index=0, key='country_key')

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
                title="Top 10 Continents by Visits",
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

                # Apply filters
                df_filtered = df.copy()
                if selected_sport_category != "All":
                    df_filtered = df_filtered[df_filtered['sports_group'] == selected_sport_category]
                if selected_device != "All":
                    df_filtered = df_filtered[df_filtered['Device'] == selected_device]
                if selected_country != "All":
                    df_filtered = df_filtered[df_filtered['Country'] == selected_country]

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
            st.title(' FunOlympics Games Dashboard')

            sport_category_options = sorted(df['sports_group'].unique())
            sport_category_options.insert(0, "All")
            selected_sport_category = st.sidebar.selectbox('Sport Category', sport_category_options, index=0, key='sport_category_key')

            device_options = sorted(df['Device'].unique())
            device_options.insert(0, "All")
            selected_device = st.sidebar.selectbox('Device', device_options, index=0, key='device_key')

            country_options = sorted(df['Full Country Name'].unique())
            country_options.insert(0, "All")
            selected_country = st.sidebar.selectbox('Country', country_options, index=0, key='country_key')


        # Apply filters
        df_filtered = df.copy()
        if selected_sport_category != "All":
            df_filtered = df_filtered[df_filtered['sports_group'] == selected_sport_category]
        if selected_device != "All":
            df_filtered = df_filtered[df_filtered['Device'] == selected_device]
        if selected_country != "All":
            df_filtered = df_filtered[df_filtered['Country'] == selected_country]
            
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
        import streamlit as st
        import pandas as pd
        import joblib
        import numpy as np

        # Data View content
        # Process new data (initial data with past 11 days)
        df = filter_data(process_data(API_TOKEN, URL))

        # Sidebar
        with st.sidebar:
            st.title(' FunOlympics Games Dashboard')

            sport_category_options = sorted(df['sports_group'].unique())
            sport_category_options.insert(0, "All")
            selected_sport_category = st.sidebar.selectbox('Sport Category', sport_category_options, index=0, key='sport_category_key')

            device_options = sorted(df['Device'].unique())
            device_options.insert(0, "All")
            selected_device = st.sidebar.selectbox('Device', device_options, index=0, key='device_key')

            country_options = sorted(df['Full Country Name'].unique())
            country_options.insert(0, "All")
            selected_country = st.sidebar.selectbox('Country', country_options, index=0, key='country_key')


        # Apply filters
        df_filtered = df.copy()
        if selected_sport_category != "All":
            df_filtered = df_filtered[df_filtered['sports_group'] == selected_sport_category]
        if selected_device != "All":
            df_filtered = df_filtered[df_filtered['Device'] == selected_device]
        if selected_country != "All":
            df_filtered = df_filtered[df_filtered['Country'] == selected_country]
            


        # Load the trained model
        best_model = joblib.load('best_gradient_boosting_model.pkl')

        # Load the label encoders
        label_encoders = joblib.load('label_encoders.pkl')

        # Example DataFrame to map country, region, and city
        # This should be replaced with your actual data

        # Function to encode the input data
        def encode_input(data, label_encoders):
            encoded_data = {}
            for column, le in label_encoders.items():
                if column in data:
                    encoded_data[column] = le.transform([data[column]])[0]
            return encoded_data


        # Streamlit app
        st.title("Predict Views")

        # Instructions
        st.write("""
        ### Welcome to the Predict Views Tool
        Please fill in the details below to get the predicted number of views for your event.
        """)

        # Event Information
        st.header("Event Information")
        resource = st.selectbox("Select Resource", label_encoders['resource'].classes_)
        event = st.selectbox("Select Event", label_encoders['Event'].classes_)
        sports_group = st.selectbox("Select Sports Group", label_encoders['sports_group'].classes_)
        resource_group = st.selectbox("Select Resource Group", label_encoders['resource_group'].classes_)
        status_code = st.selectbox("Select Status Code", label_encoders['status_code'].classes_)
        

        # User Device Information
        st.header("User Device Information")
        device = st.selectbox("Select Device", label_encoders['Device'].classes_)
        operating_system = st.selectbox("Select Operating System", label_encoders['Operating_System'].classes_)
        browser = st.selectbox("Select Browser", label_encoders['Browser'].classes_)

        # Session Information
        st.header("Session Information")
        hour = st.slider("Hour of the Day", 0, 23, 12)
        day = st.slider("Day of the Month", 1, 31, 1)
        month = st.slider("Month", 1, 12, 1)
        year = st.slider("Year", 2020, 2030, 2024)
        elapsed_time = st.number_input("Elapsed Time (ms)", min_value=0, step=1, value=0)
        session_duration = st.number_input("Session Duration (ms)", min_value=0, step=1, value=0)

        # Dynamic Dropdowns for Country, Region, and City
        st.header("Location Information")
        full_country = st.selectbox("Select Country", label_encoders['Full Country Name'].classes_)
        region_options = df_filtered[df_filtered['Full Country Name'] == full_country]['Region'].unique()
        region = st.selectbox("Select Region", label_encoders['Region'].classes_)
        city_options = df_filtered[(df_filtered['Full Country Name'] == full_country) & (df_filtered['Region'] == region)]['City'].unique()
        city = st.selectbox("Select City", label_encoders['City'].classes_)

        # Prepare the input data
        input_data = {
            'resource': resource,
            'Event': event,
            'sports_group': sports_group,
            'resource_group': resource_group,
            'status_code': status_code,
            'Full Country Name': full_country,
            'Region': region,
            'City': city,
            'Device': device,
            'Operating_System': operating_system,
            'Browser': browser,
            'hour': hour,
            'day': day,
            'month': month,
            'year': year,
            'elapsed_time': elapsed_time,
            'session_duration': session_duration
        }

        # Encode the input data
        encoded_data = {}
        for col, le in label_encoders.items():
            if col in input_data:
                encoded_data[col] = le.transform([input_data[col]])[0]

        # Convert to DataFrame
        input_df = pd.DataFrame([encoded_data])

        # Make prediction
        if st.button("Predict Views"):
            prediction = best_model.predict(input_df)
            st.write(f"Predicted Views: {int(prediction[0])}")
