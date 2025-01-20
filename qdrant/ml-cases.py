import pandas as pd
import toml
from pydantic import config

# Load Qdrant
with open("config.toml", "r") as f:
    config = toml.load(f)
# Load dataset
df = pd.read_csv("ml-cases.csv")

df.head()

df["TitleAndDescription"] = df["Title"] + " - " + df["Short Description (< 5 words)"]

df["Year"].value_counts()
df_2024 = df[df["Year"] == 2024]
df_until_2023 = df[df["Year"] < 2024]
pd.Series(df_until_2023["Year"]).value_counts()


from langchain.embeddings import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

metadatas = []
for i, row in df_until_2023.iterrows():
    metadatas.append(
        {
            "Company": row["Company"],
            "Industry": row["Industry"],
            "Tag": row["Tag"],
            "Year": row["Year"],
            "Link": row["Link"],
        }
    )

df_until_2023["TitleAndDescription"] = df_until_2023["TitleAndDescription"].astype(str)

from langchain_qdrant import QdrantVectorStore

qdrant = QdrantVectorStore.from_texts(
    texts=pd.Series(df_until_2023["TitleAndDescription"]).to_list(),
    embedding=embedding_function,
    metadatas=metadatas,
    url=config["Qdrant"]["URL"],
    api_key=config["Qdrant"]["API_KEY"],
    prefer_grpc=True,
)

question = "What kind of frauds Blablacar use machine learning to prevent?"
results = qdrant.similarity_search(question, k=5)
for i, case in enumerate(results):
    print(f"Case {i+1}:")
    print(f"{case.page_content}")
    print("====================================================")


from langchain_google_genai import ChatGoogleGenerativeAI

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    api_key=config["Gemini"]["API_KEY"],
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}"""
)

output_parser = StrOutputParser()

chain = prompt | llm_gemini | output_parser

query = "What kind of frauds Blablacar use machine learning to prevent?"
query = "What Romie can do for you?"

results = qdrant.similarity_search(query, k=5)
print("Retrieved related content :")
print(results[0].page_content)
print(results[1].page_content)
print(results[2].page_content)
print(results[3].page_content)
print(results[4].page_content)
print("====================================================")

llm_result = chain.invoke(
    {
        "input": query,
        "context": [results[0], results[1], results[2], results[3], results[4]],
    }
)

print("Question: ", query)
print("LLM Answer: ", llm_result)

# Add the new case to the dataset

from uuid import uuid4

metadatas = []
for i, row in df_2024.iterrows():
    metadatas.append(
        {
            "Company": row["Company"],
            "Industry": row["Industry"],
            "Tag": row["Tag"],
            "Year": row["Year"],
            "Link": row["Link"],
        }
    )

df_2024["TitleAndDescription"] = df_2024["TitleAndDescription"].astype(str)

uuids = [str(uuid4()) for _ in range(len(df_2024))]

qdrant.add_texts(
    texts=pd.Series(df_2024["TitleAndDescription"]).to_list(),
    metadatas=metadatas,
    ids=uuids,
)

query = "What Romie can do for you?"

results = qdrant.similarity_search(query, k=5)
print("Retrieved related content :")
print(results[0].page_content)
print(results[1].page_content)
print(results[2].page_content)
print(results[3].page_content)
print(results[4].page_content)
print("====================================================")

llm_result = chain.invoke(
    {
        "input": query,
        "context": [results[0], results[1], results[2], results[3], results[4]],
    }
)

print("Question: ", query)
print("LLM Answer: ", llm_result)
