# pdf ???
import toml
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings


# Load configuration
def load_config(config_file="config.toml"):
    with open(config_file, "r") as f:
        config = toml.load(f)
    return config


# Set up HuggingFace embeddings
def get_huggingface_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Set up Google Gemini embeddings
def get_google_gemini_embeddings(config):
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=config["Gemini"]["API_KEY"],
    )


# Set up Azure OpenAI embeddings
def get_azure_openai_embeddings(config):
    return AzureOpenAIEmbeddings(
        azure_deployment=config["AzureOpenAI"]["Embedding_DEPLOYMENT_NAME"],
        api_version=config["AzureOpenAI"]["VERSION"],
        api_key=config["AzureOpenAI"]["KEY"],
        azure_endpoint=config["AzureOpenAI"]["ENDPOINT"],
    )


# Set up the prompt chain for answering questions
def get_prompt_chain(config):
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}"""
    )
    output_parser = StrOutputParser()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", api_key=config["Gemini"]["API_KEY"]
    )
    return prompt | llm | output_parser


# Main function to run the entire process
def main():
    # Load configuration
    config = load_config()

    # Load documents
    print("Choose imput dataset")
    print("1. ./state_of_the_union.txt")
    print("2. ./rent_contract.docx")
    dataset_choice = input()
    if dataset_choice == "1":
        loader = TextLoader("state_of_the_union.txt", autodetect_encoding=True)
        query = "What did the president say about Ketanji Brown Jackson?"
    elif dataset_choice == "2":
        loader = Docx2txtLoader("rent_contract.docx")
        query = "如果我想終止租約，我應該要多久以前通知房東？"
        # ueries = [
        #     "如果我想終止租約，我應該要多久以前通知房東？",
        #     "租房子簽約時，應該要帶什麼證件？",
        #     "房間裡可以放鞭炮嗎？",
        # ]
        # print("choose one of the query")
        # for i, q in enumerate(queries):
        #     print(f"{i + 1}. {q}")
        # index = int(input())
        # query = queries[index - 1]
    else:
        raise Exception()

    chunk_size = 1000
    chunk_overlap = 0
    print("Loading documents...")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = loader.load()
    documents = text_splitter.split_documents(documents)

    # Ask user to choose a model
    print("Choose an embedding model:")
    print("1. HuggingFace (all-MiniLM-L6-v2)")
    print("2. Google Gemini (Text-Embedding-004)")
    print("3. Azure OpenAI (Text-Embedding-Ada-002)")

    choice = input("Enter the number of your choice (1, 2, or 3): ")

    if choice == "1":
        embeddings = get_huggingface_embeddings()
        model_name = "HuggingFace"
    elif choice == "2":
        embeddings = get_google_gemini_embeddings(config)
        model_name = "Google Gemini"
    elif choice == "3":
        embeddings = get_azure_openai_embeddings(config)
        model_name = "Azure OpenAI"
    else:
        print("Invalid choice. Defaulting to HuggingFace.")
        embeddings = get_huggingface_embeddings()
        model_name = "HuggingFace"

    db = FAISS.from_documents(documents, embeddings)

    # Perform similarity search using the selected model
    results = db.similarity_search_with_score(query, 2)
    print(f"Query: {query}")
    print(f"{model_name} Search Results:  ")
    print(f"Best match: {results[0][0].page_content}")
    # print(f"Second best match: {results[0][0].page_content}")

    use_chain = input("use chain? [y/n] ")
    if use_chain:
        chain = get_prompt_chain(config)
        llm_result = chain.invoke({"input": query, "context": [results[0][0]]})
        print("Chain search result:", llm_result)


if __name__ == "__main__":
    main()
