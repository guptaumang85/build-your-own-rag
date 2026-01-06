import json
import numpy as np
import pandas as pd
from pprint import pprint as original_pprint
from dateutil import parser
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os 
from together import Together
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

EMBEDDINGS = joblib.load("embeddings.joblib")

def pprint(*args, **kwargs):
    print(json.dumps(*args, indent = 2))

def format_date(date_string):
    # Parse the input string into a datetime object
    date_object = parser.parse(date_string)
    # Format the date to "YYYY-MM-DD"
    formatted_date = date_object.strftime("%Y-%m-%d")
    return formatted_date

# Read the CSV without parsing dates

def read_dataframe(path):
    df = pd.read_csv(path)

    # Apply the custom date formatting function to the relevant columns
    df['published_at'] = df['published_at'].apply(format_date)
    df['updated_at'] = df['updated_at'].apply(format_date)

    # Convert the DataFrame to dictionary after formatting
    df= df.to_dict(orient='records')
    return df

def generate_with_single_input(prompt: str, 
                               role: str = 'assistant', 
                               top_p: float = None, 
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                               together_api_key = None,
                              **kwargs):
    
    if top_p is None:
        top_p = 'none'
    if temperature is None:
        temperature = 'none'

    payload = {
            "model": model,
            "messages": [{'role': role, 'content': prompt}],
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
                  }

    if together_api_key is None:
        together_api_key = os.environ['TOGETHER_API_KEY']
    if top_p == 'none':
        payload['top_p'] = None
    if temperature == 'none':
        payload['temperature'] = None
    client = Together(api_key =  together_api_key)
    json_dict = client.chat.completions.create(**payload).model_dump()
    json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict

def concatenate_fields(dataset, fields):
    # Initialize the list where the texts will be stored    
    concatenated_data = [] 

    # Iterate over movies
    for data in dataset:
        # Initialize text as an empty string
        text = "" 

        # Iterate over the fields
        for field in fields: 
            # Get the desired field (if the key is missing an empty string should be used)
            context = data.get(field, '') 

            if context:
                # Add the context to the text (add an extra space so fields are separate)
                text += f"{context} " 

        # Strip whitespaces from the text
        text = text.strip()[:493]
        # Append the text with extra context to the list
        concatenated_data.append(text) 
    
    return concatenated_data

NEWS_DATA = pd.read_csv("./news_data_dedup.csv").to_dict(orient = 'records')

def retrieve(query, top_k = 5):
    query_embedding = model.encode(query)

    similarity_scores = cosine_similarity(query_embedding.reshape(1,-1), EMBEDDINGS)[0]
    
    similarity_indices = np.argsort(-similarity_scores)

    top_k_indices = similarity_indices[:top_k]

    return top_k_indices
