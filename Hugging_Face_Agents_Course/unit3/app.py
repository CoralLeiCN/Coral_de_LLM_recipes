# Import necessary libraries
import datasets
from langchain.docstore.document import Document
from retriever import GuestInfoRetrieverTool
from smolagents import CodeAgent, DuckDuckGoSearchTool

# Import our custom tools from their modules
from tools import HubStatsTool, WeatherInfoTool
from utils import gemini_model_OpenAIServer


# Initialize the Hugging Face model
model = gemini_model_OpenAIServer("gemini-2.0-flash")

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

# Initialize the tool
search_tool = DuckDuckGoSearchTool()
weather_info_tool = WeatherInfoTool()
hub_stats_tool = HubStatsTool()

# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3,  # Enable planning every 3 steps
)

query = "Tell me about 'Lady Ada Lovelace'"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)

query = "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)

query = "One of our guests is from Qwen. What can you tell me about their most popular model?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)


query = "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
