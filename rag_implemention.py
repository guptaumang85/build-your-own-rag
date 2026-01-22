from utils import (
    retrieve, 
    pprint, 
    generate_with_single_input, 
    read_dataframe,
    cosine_similarity
)

import joblib
import numpy as np
import bm25s
from sentence_transformers import SentenceTransformer

NEWS_DATA = read_dataframe("news_data_dedup.csv")
pprint(NEWS_DATA[9:11])

# Load the pre-computed embeddings with joblib
EMBEDDINGS = joblib.load("embeddings.joblib")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def query_news(indices):
    """
    Retrieves elements from a dataset based on specified indices.
    Parameters:
    indices (list of int): A list containing the indices of the desired elements in the dataset.
    dataset (list or sequence): The dataset from which elements are to be retrieved. It should support indexing.

    Returns:
    list: A list of elements from the dataset corresponding to the indices provided in list_of_indices.
    """
    output = [NEWS_DATA[index] for index in indices]
    return output

indices = [3, 6, 9]
pprint(query_news(indices))

indices = retrieve("Concerts in North America", top_k = 1)
print(indices)

retrieved_documents = query_news(indices)
pprint(retrieved_documents)

def get_relevant_data(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve and return the top relevant data items based on a given query.

    This function performs the following steps:
    1. Retrieves the indices of the top 'k' relevant items from a dataset based on the provided `query`.
    2. Fetches the corresponding data for these indices from the dataset.

    Parameters:
    - query (str): The search query string used to find relevant items.
    - top_k (int, optional): The number of top items to retrieve. Default is 5.

    Returns:
    - list[dict]: A list of dictionaries containing the data associated
      with the top relevant items.
    """
    # Retrieve the indices of the top_k relevant items given the query
    relevant_indices = retrieve(query, top_k)
    # Obtain the data related to the items using the indices from the previous step
    relevant_data = query_news(relevant_indices)
    return relevant_data

query = "Greatest storms in the US"
relevant_data = get_relevant_data(query, top_k = 1)
pprint(relevant_data)

def format_relevant_data(relevant_data):
    """
    Retrieves the top_k most relevant documents based on a given query and constructs an augmented prompt for a RAG system.

    Parameters:
    relevant_data (list): A list with relevant data.

    Returns:
    str: An augmented prompt with the top_k relevant documents, formatted for use in a Retrieval-Augmented Generation (RAG) system."
    """
    # Create a list so store the formatted documents
    formatted_documents = []
    # Iterates over each relevant document.
    for document in relevant_data:
        # Formats each document into a structured layout string. Remember that each document is in one different line. So you should add a new line character after each document added.
        formatted_document = f"Title: {document['title']}, Description: {document['description']}, Published at: {document['published_at']}\nURL: {document['url']}"
        # Append the formatted document string to the formatted_documents list
        formatted_documents.append(formatted_document)
    # Returns the final augmented prompt string.
    return "\n".join(formatted_documents)

example_data = NEWS_DATA[4:8]
print(format_relevant_data(example_data))

corpus = [x['title'] + " " + x['description'] for x in NEWS_DATA]
BM25_RETRIEVER = bm25s.BM25(corpus=corpus)
TOKENIZED_DATA = bm25s.tokenize(corpus)
BM25_RETRIEVER.index(TOKENIZED_DATA)

def bm25_retrieve(query: str, top_k: int = 5):
    """
    Retrieves the top k relevant documents for a given query using the BM25 algorithm.

    This function tokenizes the input query and uses a pre-indexed BM25 retriever to
    search through a collection of documents. It returns the indices of the top k documents
    that are most relevant to the query.

    Args:
        query (str): The search query for which documents need to be retrieved.
        top_k (int): The number of top relevant documents to retrieve. Default is 5.

    Returns:
        List[int]: A list of indices corresponding to the top k relevant documents
        within the corpus.
    """
    # Tokenize the query using the 'tokenize' function from the 'bm25s' module
    tokenized_query = bm25s.tokenize(query)
    
    # Use the 'BM25_RETRIEVER' to retrieve documents and their scores based on the tokenized query
    # Retrieve the top 'k' documents
    results, scores = BM25_RETRIEVER.retrieve(tokenized_query, k = top_k)

    # Extract the first element from 'results' to get the list of retrieved documents
    results = results[0]

    # Convert the retrieved documents into their corresponding indices in the results list
    top_k_indices = [corpus.index(result) for result in results]
    
    return top_k_indices

# Output is a list of indices
bm25_retrieve("What are the recent news about GDP?")
# [752, 673, 289, 626, 43]

def semantic_search_retrieve(query, top_k=5):
    """
    Retrieves the top k relevant documents for a given query using semantic search and cosine similarity.

    This function generates an embedding for the input query and compares it against pre-computed document
    embeddings using cosine similarity. The indices of the top k most similar documents are returned.

    Args:
        query (str): The search query for which relevant documents need to be retrieved.
        top_k (int): The number of top relevant documents to retrieve. Default value is 5.

    Returns:
        List[int]: A list of indices corresponding to the top k most relevant documents in the corpus.
    """
    ### START CODE HERE ###
    # Generate the embedding for the query using the pre-trained model
    query_embedding = model.encode(query)
    
    # Calculate the cosine similarity scores between the query embedding and the pre-computed document embeddings
    similarity_scores = cosine_similarity(query_embedding, EMBEDDINGS)
    
    # Sort the similarity scores in descending order and get the indices
    similarity_indices = np.argsort(-similarity_scores)

    # Select the indices of the top k documents as a numpy array
    top_k_indices_array = similarity_indices[:top_k]

    ### END CODE HERE ###
    
    # Cast them to int 
    top_k_indices = [int(x) for x in top_k_indices_array]
    
    return top_k_indices

# Let's see an example
semantic_search_retrieve("What are the recent news about GDP?")
# [743, 673, 626, 752, 326]

def reciprocal_rank_fusion(list1, list2, top_k=5, K=60):
    """
    Fuse rank from multiple IR systems using Reciprocal Rank Fusion.

    Args:
        list1 (list[int]): A list of indices of the top-k documents that match the query.
        list2 (list[int]): Another list of indices of the top-k documents that match the query.
        top_k (int): The number of top documents to consider from each list for fusion. Defaults to 5.
        K (int): A constant used in the RRF formula. Defaults to 60.

    Returns:
        list[int]: A list of indices of the top-k documents sorted by their RRF scores.
    """

    # Create a dictionary to store the RRF scores for each document index
    rrf_scores = {}

    # Iterate over each document list
    for lst in [list1, list2]:
        # Calculate the RRF score for each document index
        for rank, item in enumerate(lst, start=1): # Start = 1 set the first element as 1 and not 0. 
                                                   # This is a convention on how ranks work (the first element in ranking is denoted by 1 and not 0 as in lists)
            # If the item is not in the dictionary, initialize its score to 0
            if item not in rrf_scores:
                rrf_scores[item] = 0
            # Update the RRF score for each document index using the formula 1 / (rank + K)
            rrf_scores[item] += 1/(K+rank)

    # Sort the document indices based on their RRF scores in descending order
    sorted_items = sorted(rrf_scores, key=rrf_scores.get, reverse = True)

    # Slice the list to get the top-k document indices
    top_k_indices = [int(x) for x in sorted_items[:top_k]]

    return top_k_indices

list1 = semantic_search_retrieve('What are the recent news about GDP?')
list2 = bm25_retrieve('What are the recent news about GDP?')
rrf_list = reciprocal_rank_fusion(list1, list2)
print(f"Semantic Search List: {list1}")
print(f"BM25 List: {list2}")
print(f"RRF List: {rrf_list}")

# Output example
# Semantic Search List: [743, 673, 626, 752, 326]
# BM25 List: [752, 673, 289, 626, 43]
# RRF List: [673, 752, 626, 743, 289]

def generate_final_prompt(query, top_k, retrieve_function = None, use_rag=True):
    """
    Generates an augmented prompt for a Retrieval-Augmented Generation (RAG) system by retrieving the top_k most 
    relevant documents based on a given query.

    Parameters:
    query (str): The search query for which the relevant documents are to be retrieved.
    top_k (int): The number of top relevant documents to retrieve.
    retrieve_function (callable): The function used to retrieve relevant documents. If 'reciprocal_rank_fusion', 
                                  it will combine results from different retrieval functions.
    use_rag (bool): A flag to determine whether to incorporate retrieved data into the prompt (default is True).

    Returns:
    str: A prompt that includes the top_k relevant documents formatted for use in a RAG system.
    """

    # Define the prompt as the initial query
    prompt = query
    
    # If not using rag, return the prompt
    if not use_rag:
        return prompt


    # Determine which retrieve function to use based on its name.
    if retrieve_function.__name__ == 'reciprocal_rank_fusion':
        # Retrieve top documents using two different methods.
        list1 = semantic_search_retrieve(query, top_k)
        list2 = bm25_retrieve(query, top_k)
        # Combine the results using reciprocal rank fusion.
        top_k_indices = retrieve_function(list1, list2, top_k)
    else:
        # Use the provided retrieval function.
        top_k_indices = retrieve_function(query=query, top_k=top_k)
    
    
    # Retrieve documents from the dataset using the indices.
    relevant_documents = query_news(top_k_indices)
    
    formatted_documents = []

    # Iterate over each retrieved document.
    for document in relevant_documents:
        # Format each document into a structured string.
        formatted_document = (
            f"Title: {document['title']}, Description: {document['description']}, "
            f"Published at: {document['published_at']}\nURL: {document['url']}"
        )
        # Append the formatted string to the main data string with a newline for separation.
        formatted_documents.append(formatted_document)

    retrieve_data_formatted = "\n".join(formatted_documents)
    
    prompt = (
        f"Answer the user query below. There will be provided additional information for you to compose your answer. "
        f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
        f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
        f"Query: {query}\n"
        f"2024 News: {retrieve_data_formatted}"
    )

    
    return prompt

print(generate_final_prompt("Tell me about the US GDP in the past 3 years."))

def llm_call(query, retrieve_function = None, top_k = 5,use_rag = True):

    # Get the system and user dictionaries
    prompt = generate_final_prompt(query, top_k = top_k, retrieve_function = retrieve_function, use_rag = use_rag)

    generated_response = generate_with_single_input(prompt)

    generated_message = generated_response['content']
    
    return generated_message

query = "Tell me about the US GDP in the past 3 years."

print(llm_call(query, use_rag = True))
print(llm_call(query, use_rag = False))

query = "Recent news in technology. Provide sources."
print(llm_call(query, retrieve_function = semantic_search_retrieve))

# Output example:
# Based on the recent news in technology from 2024, here are some key points:

# 1. **Artificial Intelligence (AI) Impact on the Chip Industry**: The rapid advancement of AI is transforming the semiconductor sector, creating new winners and losers. Companies are fighting for dominance in the supply chain, and the industry is witnessing a "Game of Thrones" scenario. (Source: El Pais, April 12, 2024)

# 2. **Tech Spending Challenges for Advertising Companies**: The slower pace of business in the technology sector has continued to affect some ad holding companies in the first quarter. However, there might be a positive turn in the coming months. (Source: The Wall Street Journal, April 26, 2024)

# 3. **Market Talks in Technology, Media, and Telecom**: Recent market talks have covered various companies, including China Telecom, Bilibili, T-Mobile, Imax, and Rogers Communications. These discussions provide insights into the latest trends and developments in the industry. (Source: The Wall Street Journal, April 26 and 25, 2024)

# 4. **Energy and Utilities Roundup**: The latest market talks in the energy and utilities sector have provided updates on various companies and trends in the industry. (Source: The Wall Street Journal, April 26, 2024)

# These news articles highlight the ongoing developments and challenges in the technology sector, including the impact of AI, tech spending, and market trends in various industries.

# Sources:
# - https://www.wsj.com/articles/tech-media-telecom-roundup-market-talk-c2ae6c7a
# - https://english.elpais.com/technology/2024-04-12/artificial-intelligence-sparks-game-of-thrones-in-the-chip-industry.html
# - https://www.wsj.com/articles/tech-spending-still-proves-thorny-for-some-advertising-companies-5d8216f2
# - https://www.wsj.com/articles/tech-media-telecom-roundup-market-talk-f4376a81
# - https://www.wsj.com/articles/energy-utilities-roundup-market-talk-9e840f2f
