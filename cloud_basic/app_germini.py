from configparser import ConfigParser

import toml
from pydantic import SecretStr

with open("config.toml", "r") as f:
    config = toml.load(f)
    print(config)

from langchain_google_genai import ChatGoogleGenerativeAI

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", api_key=SecretStr(config["Gemini"]["API_KEY"])
)

user_input = "人生的意義是什麼？"

role_description = """
你是一個哲學家，請用繁體中文回答。
"""

messages = [
    ("system", role_description),
    ("human", user_input),
]

response_gemini = llm_gemini.invoke(messages)

print(f"問 : {user_input}")
print(f"Gemini : {response_gemini.content}")
