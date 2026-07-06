import os
from dotenv import load_dotenv

load_dotenv()

IBM_API_KEY = os.getenv("IBM_API_KEY", "your_api_key_here")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "your_project_id_here")
IBM_URL = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")
