import streamlit as st
import pandas as pd
import requests
import json
import re
import time
import os
from datetime import datetime

#Load health data from JSON file
HEALTH_DATA_PATH = "/Users/nicktran/Downloads/Integrated_Data/P01Data.json"

@st.cache_data
def load_health_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Transform data into dictionary format
        health_data = {}
        for entry in data:
            date_str = entry["dateTime"].split('T')[0]  # Extract date part
            health_data[date_str] = entry["healthDomain"]
        return health_data
    except Exception as e:
        st.error(f"Error loading health data: {e}")
        return {}
    

#Load health data
health_data_dict = load_health_data(HEALTH_DATA_PATH)
present_date = list(health_data_dict.keys())[-1]

@st.cache_data
def load_prompts(file_path):
    return pd.read_csv(file_path)
def extract_fields_with_ollama(prompt):
    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "llama3"
    system_prompt = (
        f"Assume the present date is {present_date} if the end date is implied"
        "Extract the health_ailment and the start and end dates (timeframe) from this query. "
        "Return ONLY JSON format: {\"health_ailment\": \"...\", \"start_date\": \"YYYY-MM-DD\", \"end_date\": \"YYYY-MM-DD\"}. "
        "Dates must be in the format YYYY-MM-DD (for example: 2025-07-03)."
        "If only one date is given then start_date = end_date"
        f"If the query mentions a relative timeframe (such as 'since last week', 'for the past month', 'recently', etc.), infer the start_date based on the present date and the described period, and set end_date to {present_date}. "
    )
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0, 
            "num_ctx": 4096
        }
    }
    try:
        start_time = time.time()
        response = requests.post(OLLAMA_URL, json=payload)
        response_time = time.time() - start_time
        if response.status_code == 200:
            response_data = response.json()
            if "response" in response_data:
                response_text = response_data["response"]
                try:
                    json_data = json.loads(response_text)
                    return (
                        json_data.get('health_ailment'),
                        json_data.get('start_date'),
                        json_data.get('end_date'),
                        response_time
                    )
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        json_data = json.loads(json_str)
                        return (
                            json_data.get('health_ailment'),
                            json_data.get('start_date'),
                            json_data.get('end_date'),
                            response_time
                        )
        st.error(f"Ollama response format error: {response_data}")
        return None, None, None, response_time
    except Exception as e:
        st.error(f"Ollama connection failed: {e}")
        return None, None, None, 0


st.title("Health Ailment Prompts")

#Upload prompts file
st.sidebar.header("Dataset Options")
file = st.sidebar.file_uploader("Upload Prompts CSV", type=["csv"])

if file:
    prompts_df = load_prompts(file)
else:
    st.warning("Please upload your prompts CSV file to begin.")
    st.stop()

#Check for required columns
required_columns = ['prompt', 'health_ailment', 'date']
missing = [col for col in required_columns if col not in prompts_df.columns]
if missing:
    st.error(f"Missing columns in CSV: {', '.join(missing)}")
    st.stop()

#Ollama status check
try:
    status_response = requests.get("http://localhost:11434")
    if status_response.status_code != 200:
        st.error("Ollama not running! Start Ollama with: ollama serve")
        st.stop()
except:
    st.error("Ollama not running! Start Ollama with: ollama serve")
    st.stop()

user_prompt = st.text_input(
    "Type your health query",
    value="I've been experiencing severe migraines since last May"
)

def get_dates_in_range(start_date, end_date, data_dict):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return {
        date: data_dict[date]
        for date in data_dict
        if start <= datetime.strptime(date, "%Y-%m-%d") <= end
    }

if user_prompt:
    with st.spinner("Extracting fields with Llama 3..."):
        extracted_ailment, start_date, end_date, response_time = extract_fields_with_ollama(user_prompt)
        
        if extracted_ailment and start_date and end_date:
            st.write(f"**Extracted Health Ailment:** `{extracted_ailment}`")
            st.write(f"**Extracted Timeframe:** `{start_date}` to `{end_date}`")
            st.caption(f"Extraction time: {response_time:.2f} seconds")

            #Retrieve data for the date range
            health_data_range = get_dates_in_range(start_date, end_date, health_data_dict)
            if health_data_range:
                st.success("Health data retrieved successfully for the selected timeframe!")
                st.json(health_data_range)
            else:
                st.error(f"No health data found for the timeframe: {start_date} to {end_date}")

            #Filter local DataFrame for display
            filtered_df = prompts_df[
                (prompts_df['health_ailment'].str.lower() == extracted_ailment.lower()) &
                (prompts_df['date'] >= start_date) &
                (prompts_df['date'] <= end_date)
            ]
            if not filtered_df.empty:
                st.subheader("Matching Prompts in Dataset")
                st.dataframe(filtered_df[['prompt', 'health_ailment', 'date']])
            else:
                st.warning("No matching prompts found in your CSV dataset")
        else:
            st.error("Field extraction failed. Try a different query format.")
else:
    st.info("Enter a health query above to analyze")


#Add PM Health database documentation
st.sidebar.markdown("### Health Data Source")
st.sidebar.write(f"Loaded from: `{os.path.basename(HEALTH_DATA_PATH)}`")
