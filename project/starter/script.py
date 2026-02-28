from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')

# Ensure the keys are loaded correctly
assert openai_api_key is not None, "OPENAI_API_KEY is not set"
assert tavily_api_key is not None, "TAVILY_API_KEY is not set"

print("API keys loaded successfully!")
