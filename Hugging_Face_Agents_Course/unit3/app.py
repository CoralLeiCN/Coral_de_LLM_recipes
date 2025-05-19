import random

import datasets
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from retriever import GuestInfoRetrieverTool
from smolagents import CodeAgent, DuckDuckGoSearchTool, Tool
from tools import WeatherInfoTool
from utils import gemini_model_OpenAIServer

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join(
            [
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}",
            ]
        ),
        metadata={"name": guest["name"]},
    )
    for guest in guest_dataset
]


# Initialize the tool
guest_info_tool = GuestInfoRetrieverTool(docs)


# Initialize the Hugging Face model
model = gemini_model_OpenAIServer("gemini-2.0-flash")
# Create Alfred, our gala agent, with the guest info tool
alfred = CodeAgent(tools=[guest_info_tool], model=model)

# Example query Alfred might receive during the gala
response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

print("🎩 Alfred's Response:")
print(response)


# Initialize the DuckDuckGo search tool
search_tool = DuckDuckGoSearchTool()

# Example usage
results = search_tool("Who's the current President of France?")
print(results)


# Initialize the tool
weather_info_tool = WeatherInfoTool()
