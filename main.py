import streamlit as st
import pandas as pd
import os
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import datetime

# Directory to save monthly data
data_dir = "uploaded_data"

# Create the directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

st.header('Sentiment Analysis')

# CSV analysis and saving the data
with st.expander('Upload CSV and Save Data'):
    upl = st.file_uploader('Upload CSV file')

    # Function to calculate polarity score
    def score(x):
        try:
            blob1 = TextBlob(str(x))  # Ensure input is converted to string
            return blob1.sentiment.polarity
        except Exception as e:
            return 0.0  # Handle any error in sentiment analysis gracefully

    # Function to analyze sentiment
    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        # Read the uploaded CSV
        df = pd.read_csv(upl)

        # Check if 'Unnamed: 0' exists and remove it
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']

        # Check if 'Feedback' column exists
        if 'Feedback' in df.columns:
            df['score'] = df['Feedback'].apply(score)
            df['analysis'] = df['score'].apply(analyze)
        else:
            st.error("The CSV must contain a 'Feedback' column.")

        # Extract current month and year
        current_month = datetime.now().strftime("%B-%Y")

        # Save the dataframe for the current month as CSV
        file_path = os.path.join(data_dir, f"feedback_{current_month}.csv")
        df.to_csv(file_path, index=False)
        st.write(f"Data saved for {current_month}. File: {file_path}")

        # Show the first 10 rows of the dataframe
        st.write(df.head(10))

        # Cache the dataframe conversion to CSV
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df)

        # Button to download the CSV
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f'sentiment_{current_month}.csv',
            mime='text/csv',
        )

# Function to load data for a specific month
def load_month_data(month_year):
    file_path = os.path.join(data_dir, f"feedback_{month_year}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None

# Function to list all available month-year data in the directory
def list_available_months():
    files = os.listdir(data_dir)
    months = [file.replace("feedback_", "").replace(".csv", "") for file in files if file.startswith("feedback_")]
    return months

# Define custom colors for each sentiment category
sentiment_colors = {
    'Positive': '#4CAF50',  # Green
    'Negative': '#F44336',  # Red
    'Neutral': '#9E9E9E'    # Gray
}

# Initialize session state for tracking deletion
if "deleted_month" not in st.session_state:
    st.session_state["deleted_month"] = None

# List available months with saved data
available_months = list_available_months()

# Add section to view sentiment trends across all months
with st.expander('View Sentiment Trends Across All Months'):
    if available_months:
        st.write("Available months with data:", available_months)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Select chart type for monthly trends
        monthly_chart_type = st.selectbox("Select chart type for monthly trends", ["Line", "Bar"])

        # Loop over each month and plot the sentiment distribution
        for month in available_months:
            month_data = load_month_data(month)
            if month_data is not None:
                sentiment_counts = month_data['analysis'].value_counts()

                # Plot with predefined colors
                if monthly_chart_type == "Line":
                    sentiment_counts.plot(kind='line', marker='o', ax=ax, label=month, color=[sentiment_colors.get(c, "#333333") for c in sentiment_counts.index])
                elif monthly_chart_type == "Bar":
                    sentiment_counts.plot(kind='bar', ax=ax, label=month, color=[sentiment_colors.get(c, "#333333") for c in sentiment_counts.index])

        ax.set_title('Sentiment Trends Across Months')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.legend(title="Months")
        st.pyplot(fig)
    else:
        st.write("No monthly data available yet.")

# Add section to view record of data (minimum 10 records)
with st.expander('View Record of the Data'):
    if available_months:
        selected_month = st.selectbox("Select a month to view records", available_months)
        month_data = load_month_data(selected_month)
        if month_data is not None:
            st.write(f"Displaying 10 records of {selected_month}")
            st.write(month_data.head(10))  # Display the first 10 records

# Add section to compare data by multiple months (no pie chart)
if available_months:
    with st.expander('Compare Data by Multiple Months'):
        selected_months = st.multiselect("Choose months to compare", available_months)
        if selected_months:
            all_data = pd.DataFrame()
            for month in selected_months:
                month_data = load_month_data(month)
                if month_data is not None:
                    month_data['Month'] = month  # Add a column for month
                    all_data = pd.concat([all_data, month_data], ignore_index=True)

            if not all_data.empty:
                st.write(all_data.head(10))  # Display first 10 rows of combined data

                sentiment_counts = all_data.groupby(['Month', 'analysis']).size().unstack(fill_value=0)

                chart_type = st.selectbox("Select chart type", ["Bar", "Line"])

                if chart_type == "Bar":
                    fig, ax = plt.subplots()
                    sentiment_counts.plot(kind='bar', stacked=True, ax=ax, color=[sentiment_colors.get(c, "#333333") for c in sentiment_counts.columns])
                    ax.set_title(f"Sentiment Analysis for Selected Months")
                    ax.set_xlabel("Month")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)

                elif chart_type == "Line":
                    fig, ax = plt.subplots()
                    sentiment_counts.plot(kind='line', marker='o', ax=ax, color=[sentiment_colors.get(c, "#333333") for c in sentiment_counts.columns])
                    ax.set_title(f"Sentiment Analysis for Selected Months")
                    ax.set_xlabel("Month")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
            else:
                st.write("No data available for the selected months.")

# Add delete data option
if available_months:
    with st.expander('Delete Data'):
        month_to_delete = st.selectbox("Select a month to delete", available_months)

        if st.button(f"Delete data for {month_to_delete}"):
            file_path = os.path.join(data_dir, f"feedback_{month_to_delete}.csv")
            if os.path.exists(file_path):
                os.remove(file_path)
                st.session_state["deleted_month"] = month_to_delete
                st.success(f"Data for {month_to_delete} has been deleted.")
                available_months = list_available_months()  # Refresh the list of available months
            else:
                st.error(f"File for {month_to_delete} not found.")

    if st.session_state["deleted_month"]:
        st.write(f"Deleted data for: {st.session_state['deleted_month']}")
else:
    st.write("No data available to delete.")
