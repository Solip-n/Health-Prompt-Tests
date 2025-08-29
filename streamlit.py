import streamlit as st
import pandas as pd
import requests
import json
import re
import time
import os
from datetime import datetime
from transformers import pipeline
from google import genai

@st.cache_resource
def stream_gemini_response(prompt, model="gemini-2.0-flash"):
    client = genai.Client(api_key="AIzaSyCxTS79aoXru-1WJ2RAfdCUuv3PQgwWqCE") 
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text


def build_gemini_prompt(ailment, health_data_window):
    prompt_text = f"""
    You are gemini 2.0 flash, a friendly, knowledgeable medical AI trained to assist patients in understanding their symptoms.

    A patient has shared the following:
    Health Concern: {ailment}
    health_data_window: {health_data_window}

    Based on this, provide the following:
    1. Brief Acknowledgement: Show empathy and confirm you’ve understood the concern.
    2. Reasoning & Possibilities: Explain 2–3 possible general causes for the symptom, based on its duration. Use clear reasoning, explaining why each possibility is relevant (e.g., “Since this has lasted over 2 weeks, it could be related to...”).
    3. Follow-up Questions: Ask 2–3 simple follow-up questions that could help clarify the situation.
    4. Next Steps: Offer general suggestions (like rest, hydration, tracking symptoms). Mention if this may need in-person care (e.g., “If your symptoms worsen or don’t improve in X days, please consider seeing a doctor.”).
    5. Caution: Do not give a diagnosis or suggest specific medications. Avoid alarming language; focus on support and clarity.

    Use features from the dataset in health_data_window to support your reasoning. Use simple, respectful, and patient-friendly language in your response.
    """
    return prompt_text

HEALTH_DATA_PATH = "/Users/nicktran/Downloads/pmdata/P01Data.json"

@st.cache_data
def load_health_data(file_path):
   try:
       with open(file_path, 'r') as f:
           data = json.load(f)
       health_data = {}
       for entry in data:
           dt = entry["dateTime"]
           dt_obj = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
           dt_hour = dt_obj.strftime("%Y-%m-%dT%H")  #hour precision
           if dt_hour not in health_data:
               health_data[dt_hour] = entry["healthDomain"]
       return health_data
   except Exception as e:
       st.error(f"Error loading health data: {e}")
       return {}


health_data_dict = load_health_data(HEALTH_DATA_PATH)
present_date = list(health_data_dict.keys())[-1]


def save_data_to_json(data, filename="health_data_timeframe.json"):
   with open(filename, "w") as f:
       json.dump(data, f, indent=2)


def strip_to_hour(dt_str):
   return dt_str[:13]

@st.cache_data
def load_prompts(file_path):
   return pd.read_csv(file_path)


def extract_fields_with_ollama(prompt):
   OLLAMA_URL = "http://localhost:11434/api/generate"
   MODEL_NAME = "llama3"
   system_prompt = (
       f"Assume the present date is {present_date} if the end date is implied. "
       "Extract the health_ailment and the start and end dates (timeframe) from this query. "
       "Return ONLY JSON format: {\"health_ailment\": \"...\", \"start_date\": \"%Y-%m-%dT%H\", \"end_date\": \"%Y-%m-%dT%H\"}. "
       "Dates must be in the format %Y-%m-%dT%H (for example: 2019-12-06T19). "
       "If only one date is given then start_date = end_date. "
       f"If the query mentions a relative timeframe (such as 'since last week', 'for the past month', 'recently', etc.), infer the start_date based on the present date and the described period, and set end_date to {present_date}."
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
               except json.JSONDecodeError:
                   json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                   if json_match:
                       json_str = json_match.group(0)
                       json_data = json.loads(json_str)
                   else:
                       st.error("Could not parse JSON from Ollama response.")
                       return None, None, None, response_time
               start_date = strip_to_hour(json_data.get('start_date', ''))
               end_date = strip_to_hour(json_data.get('end_date', ''))
               return (
                   json_data.get('health_ailment'),
                   start_date,
                   end_date,
                   response_time
               )
       st.error(f"Ollama response format error: {response_data}")
       return None, None, None, response_time
   except Exception as e:
       st.error(f"Ollama connection failed: {e}")
       return None, None, None, 0


st.title("Health Ailment Prompts")

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


def get_datetimes_in_range(start_datetime, end_datetime, data_dict):
   return {
       dt: data_dict[dt]
       for dt in data_dict
       if start_datetime <= dt <= end_datetime
   }


if user_prompt:
   with st.spinner("Extracting fields..."):
       extracted_ailment, start_date, end_date, response_time = extract_fields_with_ollama(user_prompt)
       if extracted_ailment and start_date and end_date:
           st.write(f"**Extracted Health Ailment:** `{extracted_ailment}`")
           st.write(f"**Extracted Timeframe:** `{start_date}` to `{end_date}`")
           st.caption(f"Extraction time: {response_time:.2f} seconds")

           #retrieve data for the date range
           health_data_range = get_datetimes_in_range(start_date, end_date, health_data_dict)
           if health_data_range:
               st.success("Health data retrieved successfully for the selected timeframe!")
               gemini_prompt = build_gemini_prompt(extracted_ailment, health_data_range)
               st.subheader("Response")
               with st.spinner("generating response..."):
                   response_placeholder = st.empty()
                   summary_so_far = ""
                   for chunk in stream_gemini_response(gemini_prompt):
                       summary_so_far += chunk
                       response_placeholder.markdown(summary_so_far)
               st.json(health_data_range)
           else:
               st.error(f"No health data found for the timeframe: {start_date} to {end_date}")

           #filter DataFrame for display
           '''
           filtered_df = prompts_df[
               (prompts_df['health_ailment'].str.lower() == extracted_ailment.lower()) &
               (prompts_df['date'].str[:13] >= start_date) &
               (prompts_df['date'].str[:13] <= end_date)
           ]

           if not filtered_df.empty:
               st.subheader("Matching Prompts in Dataset")
               st.dataframe(filtered_df[['prompt', 'health_ailment', 'date']])
           else:
               st.warning("No matching prompts found in your CSV dataset")
            '''
       else:
           st.error("Field extraction failed. Try a different query format.")
else:
   st.info("Enter a health query above to analyze")


st.sidebar.markdown("### Health Data Source")
st.sidebar.write(f"Loaded from: `{os.path.basename(HEALTH_DATA_PATH)}`")
if health_data_range:
   json_str = json.dumps(health_data_range, indent=2)
   st.sidebar.download_button(
       label="Download extracted data as JSON",
       data=json_str,
       file_name="health_data_timeframe.json",
       mime="application/json"
   )
