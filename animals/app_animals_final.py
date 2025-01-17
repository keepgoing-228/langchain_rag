import pandas as pd
import toml
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from pandas.core.algorithms import mode


def main():
    with open("config.toml", "r") as f:
        config = toml.load(f)

    # Load dataset
    animal_data = pd.read_csv("animal-fun-facts-dataset.csv")

    mode = input("1. local, 2. azure, 3. gemini? ")
    local_name = "animal-db-"
    if mode == "1":
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        local_name += "local"
    elif mode == "2":
        embedding_function = AzureOpenAIEmbeddings(
            azure_deployment=config["AzureOpenAI"]["Embedding_DEPLOYMENT_NAME"],
            api_version=config["AzureOpenAI"]["VERSION"],
            azure_endpoint=config["AzureOpenAI"]["ENDPOINT"],
            api_key=config["AzureOpenAI"]["KEY"],
        )
        local_name += "azure"
    elif mode == "3":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=config["Gemini"]["API_KEY"],
        )
        local_name += "gemini"
    else:
        raise ValueError("Invalid mode")

    # Create FAISS vector store
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

    faiss = FAISS.from_texts(
        animal_data["text"].to_list(), embedding_function, metadatas
    )

    # Query vector store
    query = "What is ship of the desert?"
    result = faiss.similarity_search_with_score(query, 5)
    print(result)

    # Save vector store
    answer = input("save vector store? [y/n]")
    if answer == "y":
        faiss.save_local(local_name, "index")


if __name__ == "__main__":
    main()
