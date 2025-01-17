from configparser import ConfigParser

import toml

# Set up the config parser
with open("config.toml", "r") as f:
    config = toml.load(f)

import pandas as pd
from langchain import FAISS

# Load dataset
animal_data = pd.read_csv("animal-fun-facts-dataset.csv")

# Embedding function - SentenceTransformer - all-MiniLM-L6-v2
# from langchain.embeddings import SentenceTransformerEmbeddings

# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Embedding function - AzureOpenAI - text-embedding-ada-002

from langchain_openai import AzureOpenAIEmbeddings

embedding_function = AzureOpenAIEmbeddings(
    azure_deployment=config["AzureOpenAI"]["Embedding_DEPLOYMENT_NAME"],
    api_version=config["AzureOpenAI"]["VERSION"],
    azure_endpoint=config["AzureOpenAI"]["ENDPOINT"],
    api_key=config["AzureOpenAI"]["KEY"],
)


metadatas = []
for i, row in animal_data.iterrows():
    metadatas.append(
        {
            "Animal Name": row["animal_name"],
            "Source URL": row["source"],
            # "Media URL": row["media_link"],
            # "Wikipedia URL": row["wikipedia_link"],
        }
    )

animal_data["text"] = animal_data["text"].astype(str)

faiss = FAISS.from_texts(animal_data["text"].to_list(), embedding_function, metadatas)

result = faiss.similarity_search_with_score("What is ship of the desert?", 3)

faiss.similarity_search_with_score("What is TibaMe?", 3)

# export the model
faiss.save_local("animal_db_openai", "index")

print(result)
