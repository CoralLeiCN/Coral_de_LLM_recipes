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

# Create Alfred with conversation memory
alfred_with_memory = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=True,
    planning_interval=3,
)

# First interaction
response1 = alfred_with_memory.run("Tell me about Lady Ada Lovelace.")
print("🎩 Alfred's First Response:")
print(response1)

# Second interaction (referencing the first)
response2 = alfred_with_memory.run(
    "What projects is she currently working on?", reset=False
)
print("🎩 Alfred's Second Response:")
print(response2)
