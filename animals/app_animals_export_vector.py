import pandas as pd
from langchain import FAISS

# Load dataset
animal_data = pd.read_csv("animal-fun-facts-dataset.csv")

from langchain.embeddings import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

metadatas = []  # for help, keep the important info
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

faiss.similarity_search_with_score("What is ship of the desert?", 3)


# export the model
faiss.save_local("animals_db", "index")
