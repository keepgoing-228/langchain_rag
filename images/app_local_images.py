import base64
from configparser import ConfigParser

import toml
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

with open("config.toml", "r") as f:
    config = toml.load(f)
    print(config)

# gemini-2.0-flash-exp
# gemini-1.5-flash-latest

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    api_key=SecretStr(config["Gemini"]["API_KEY"]),
    max_tokens=8192,
)


def image4LangChain(image_url):
    if "http" in image_url:
        return {"url": image_url}
    else:
        with open(image_url, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        return {"url": f"data:image/jpeg;base64,{image_data}"}


user_messages = []
# append user input question
user_input = "are the two images the same? Can these animals be friends?請詳細描述。"
user_messages.append({"type": "text", "text": user_input + "請使用繁體中文回答。"})
# append images
# image_url1 = "https://i.ibb.co/KyNtMw5/IMG-20240321-172354614-AE.jpg"
# image_url2 = "cat.jpg"
image_url1 = ""
image_url2 = ""

user_messages.append(
    {
        "type": "image_url",
        "image_url": image4LangChain(image_url1),
    }
)

user_messages.append(
    {
        "type": "image_url",
        "image_url": image4LangChain(image_url2),
    }
)

human_messages = HumanMessage(content=user_messages)
result = llm.invoke([human_messages])

print("Q: " + user_input)
print(result.content)

# Display the image
# display(Image(url=image_url))
