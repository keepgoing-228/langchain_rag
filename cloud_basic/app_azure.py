from configparser import ConfigParser

import toml
from pydantic import SecretStr

# Set up the config parser
with open("config.toml", "r") as f:
    config = toml.load(f)
    print(config)

from langchain_openai import AzureChatOpenAI

# 1. Tell me a joke
llm_gpt4o = AzureChatOpenAI(
    api_version=config["AzureOpenAI"]["VERSION"],
    azure_deployment=config["AzureOpenAI"]["GPT4o_DEPLOYMENT_NAME"],
    azure_endpoint=config["AzureOpenAI"]["ENDPOINT"],
    api_key=SecretStr(config["AzureOpenAI"]["KEY"]),
    max_tokens=200,
)

user_input = "人生的意義是什麼？"

role_description = """
你是一個哲學家，請用繁體中文回答。
"""

messages = [
    ("system", role_description),
    ("human", user_input),
]

response_gpt4o = llm_gpt4o.invoke(messages)

print(f"問 : {user_input}")
print(f"GPT4o : {response_gpt4o.content}")


##################################################################
