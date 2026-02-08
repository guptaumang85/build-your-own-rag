from utils import (
    retrieve, 
    pprint, 
    generate_with_single_input,
    print_object_properties,
    read_dataframe,
    cosine_similarity
)

import joblib
import numpy as np
import bm25s
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.query import (
    Filter, 
    Rerank
)


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

client = weaviate.connect_to_local(port=8079, grpc_port=50050)
bbc_data = joblib.load('data/bbc_data.joblib')
collection = client.collections.get("bbc_collection") # collection has to be created in vector db.
print(f"The number of elements in the collection is: {len(collection)}")

object = collection.query.fetch_objects(limit = 1, include_vector = True).objects[0]
print("Printing the properties (some will be truncated due to size)")
print_object_properties(object.properties)
print("Vector: (truncated)",object.vector['main_vector'][0:15])
print("Vector length: ", len(object.vector['main_vector']))

def filter_by_metadata(metadata_property: str, 
                       values: list[str], 
                       collection: "weaviate.collections.collection.sync.Collection" , 
                       limit: int = 5) -> list:
    """
    Retrieves objects from a specified collection based on metadata filtering criteria.

    This function queries a collection within the specified client to fetch objects that match 
    certain metadata criteria. It uses a filter to find objects whose specified 'property' contains 
    any of the given 'values'. The number of objects retrieved is limited by the 'limit' parameter.

    Args:
    metadata_property (str): The name of the metadata property to filter on.
    values (List[str]): A list of values to be matched against the specified property.
    collection_name (weaviate.collections.collection.sync.Collection): The collection to query.
    limit (int, optional): The maximum number of objects to retrieve. Defaults to 5.

    Returns:
    List[Object]: A list of objects from the collection that match the filtering criteria.
    """
    
    response = collection.query.fetch_objects(limit=limit, filters = Filter.by_property(metadata_property).contains_any(values))
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects

res = filter_by_metadata('title', ['Taylor Swift'], collection, limit = 2)
for x in res:
    print_object_properties(x)

def semantic_search_retrieve(query: str,
                             collection: "weaviate.collections.collection.sync.Collection" , 
                             top_k: int = 5) -> list:
    """
    Performs a semantic search on a collection and retrieves the top relevant chunks.

    This function executes a semantic search query on a specified collection to find text chunks 
    that are most relevant to the input 'query'. The search retrieves a limited number of top 
    matching objects, as specified by 'top_k'. The function returns the 'chunk' property of 
    each of the top matching objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the semantic search is performed.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """

    response = collection.query.near_text(query = query, limit = top_k)
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects

print_object_properties(semantic_search_retrieve(query = 'Tell me about the last Taylor Swift show', collection = collection, top_k = 2))

def bm25_retrieve(query: str, 
                  collection: "weaviate.collections.collection.sync.Collection" , 
                  top_k: int = 5) -> list:
    """
    Performs a BM25 search on a collection and retrieves the top relevant chunks.

    This function executes a BM25-based search query on a specified collection to identify text 
    chunks that are most relevant to the provided 'query'. It retrieves a limited number of the 
    top matching objects, as specified by 'top_k', and returns the 'chunk' property of these objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the BM25 search is performed.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """

    response = collection.query.bm25(query=query, limit=top_k)
    
    response_objects = [x.properties for x in response.objects]
    return response_objects

print_object_properties(bm25_retrieve('Tell me about the last Taylor Swift show', collection, top_k = 2))

def hybrid_retrieve(query: str, 
                    collection: "weaviate.collections.collection.sync.Collection" , 
                    alpha: float = 0.5,
                    top_k: int = 5
                   ) -> list:
    """
    Performs a hybrid search on a collection and retrieves the top relevant chunks.

    This function executes a hybrid search that combines semantic vector search and traditional 
    keyword-based search on a specified collection to find text chunks most relevant to the 
    input 'query'. The relevance of results is influenced by 'alpha', which balances the weight 
    between vector and keyword matches. It retrieves a limited number of top matching objects, 
    as specified by 'top_k', and returns the 'chunk' property of these objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the hybrid search is performed.
    alpha (float, optional): A weighting factor that balances the contribution of semantic 
    and keyword matches. Defaults to 0.5.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """
    response = collection.query.hybrid(query=query, alpha=alpha, limit=top_k)
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects 

print_object_properties(hybrid_retrieve('Tell me about the last Taylor Swift show', collection, top_k = 2))

def semantic_search_with_reranking(query: str, 
                                   rerank_property: str,
                                   collection: "weaviate.collections.collection.sync.Collection" , 
                                   rerank_query: str = None,
                                   top_k: int = 5
                                   ) -> list:
    """
    Performs a semantic search and reranks the results based on a specified property.

    Args:
        query (str): The search query to perform the initial search.
        rerank_property (str): The property used for reranking the search results.
        collection (weaviate.collections.collection.sync.Collection): The collection to search within.
        rerank_query (str, optional): The query to use specifically for reranking. If not provided, 
                                      the original query is used for reranking.
        top_k (int, optional): The maximum number of top results to return. Defaults to 5.

    Returns:
        list: A list of properties from the reranked search results, where each item corresponds to 
              an object in the collection.
    """

    # Set the rerank_query to be the same as the query if rerank_query is not passed (don't change this line)
    if rerank_query is None: 
        rerank_query = query 
        
    # Define the reranker with rerank_query and rerank_property
    reranker = Rerank(
            prop=rerank_property,                   # The property to rerank on
            query=rerank_query  # If not provided, the original query will be used
        )

    # Retrieve using collection.query.near_text with the appropriate parameters (do not forget the rerank!)
    response = collection.query.near_text(query=query, rerank=reranker, limit=top_k)
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects 

query = 'Tell me about the conflicts in Latin America'
# Get the results from a search (in this case the hybrid search)
results = semantic_search_with_reranking(query, collection = collection, top_k = 2, rerank_property = 'chunk')
print_object_properties(results)

def generate_final_prompt(query: str, 
                          top_k: int, 
                          retrieve_function: callable,
                          rerank_query: str = None, 
                          rerank_property: str = None, 
                          use_rerank: bool = False, 
                          use_rag: bool = True) -> str:
    """
    Generates a final prompt by optionally retrieving and formatting relevant documents using retrieval-augmented generation (RAG).

    Args:
        query (str): The initial query to be used for document retrieval.
        top_k (int): The number of top documents to retrieve and use for generating the prompt.
        retrieve_function (callable): The function used to retrieve documents based on the query.
        rerank_query (str, optional): The query used specifically for reranking documents if reranking is enabled.
        rerank_property (str, optional): The property used for reranking. Required if 'use_rerank' is True.
        use_rerank (bool, optional): Flag to denote whether to use reranking in document retrieval. Defaults to False.
        use_rag (bool, optional): Flag to determine whether to use retrieval-augmented generation. Defaults to True.

    Returns:
        str: A constructed prompt that includes the original query and formatted retrieved documents if 'use_rag' is True.
             Otherwise, it returns the original query.
    """
    # If no rag, return the query
    if not use_rag:
        return query
    
    if use_rerank:
        if rerank_property is None:
            raise ValueError('rerank_property must be set if use_rerank = True')
        top_k_documents = retrieve_function(query=query, top_k=top_k, collection = collection, rerank_property = rerank_property, rerank_query = rerank_query)
    else:
        top_k_documents = retrieve_function(query=query, top_k=top_k, collection = collection)
    
    # Initialize an empty string to store the formatted data.
    formatted_data = ""
    
    # Iterate over each retrieved document.
    for document in top_k_documents:
        # Format each document into a structured string.
        document_layout = (
            f"Title: {document['title']}, Chunk: {document['chunk']}, "
            f"Published at: {document['pubDate']}\nURL: {document['link']}"
        )
        # Append the formatted string to the main data string with a newline for separation.
        formatted_data += document_layout + "\n"
    
    # If use_rag flag is True, construct the enhanced prompt with the augmented data.
    retrieve_data_formatted = formatted_data  # Store formatted data.
    prompt = (
        f"Answer the user query below. There will be provided additional information for you to compose your answer. "
        f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
        f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
        f"The news data is ordered by relevance."
        f"Query: {query}\n"
        f"2024 News: {retrieve_data_formatted}"
    )
    
    return prompt

print(generate_final_prompt("Tell me about the US GDP in the past 3 years."))

prompt = generate_final_prompt("Tell me the economic situation of the US in 2024.", top_k = 5, retrieve_function = semantic_search_retrieve, use_rerank = False, rerank_property = 'title')
print(prompt)

def llm_call(query: str, 
             retrieve_function: callable = None, 
             top_k: int = 5, 
             use_rag: bool = True, 
             use_rerank: bool = False, 
             rerank_property: str = None, 
             rerank_query: str = None) -> str:
    """
    Simulates a call to a language model by generating a prompt and using it to produce a response.

    Args:
        query (str): The initial query for which a response is sought.
        retrieve_function (callable, optional): The function used to retrieve documents related to the query.
        top_k (int, optional): The number of top documents to retrieve and use for generating the prompt. Defaults to 5.
        use_rag (bool, optional): Indicates whether to use retrieval-augmented generation. Defaults to True.
        use_rerank (bool, optional): Indicates whether to apply reranking to the retrieved documents. Defaults to False.
        rerank_property (str, optional): The property to use for reranking. Required if 'use_rerank' is True.
        rerank_query (str, optional): The query used specifically for reranking documents if reranking is enabled.

    Returns:
        str: The generated response content after processing the prompt with a language model.
    """
    
    # Get the prompt
    PROMPT = generate_final_prompt(query, top_k = top_k, retrieve_function = retrieve_function, use_rag = use_rag, use_rerank = use_rerank, rerank_property = rerank_property, rerank_query = rerank_query)
    
    generated_response = generate_with_single_input(PROMPT)

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


query = "Tell me about United States and Brazil's relationship over the course of 2024. Provide links for the resources you use in the answer."

print(llm_call(query = query, 
               top_k = 5, 
               retrieve_function = hybrid_retrieve, 
               ))