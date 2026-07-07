import os
import sys
from dotenv import load_dotenv

load_dotenv()

IBM_API_KEY = os.getenv("IBM_API_KEY", "your_api_key_here")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "your_project_id_here")
IBM_URL = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")
IBM_PRIMARY_MODEL = os.getenv("IBM_PRIMARY_MODEL", "meta-llama/llama-3-3-70b-instruct")
IBM_FALLBACK_MODEL = os.getenv("IBM_FALLBACK_MODEL", "meta-llama/llama-3-1-8b")
IBM_EMBEDDING_MODEL = os.getenv("IBM_EMBEDDING_MODEL", "sentence-transformers/all-minilm-l6-v2")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

if IBM_API_KEY == "your_api_key_here" or not IBM_API_KEY:
    logger.critical("IBM_API_KEY is not set. Please set it in your environment variables or .env file.")
    sys.exit(1)
if IBM_PROJECT_ID == "your_project_id_here" or not IBM_PROJECT_ID:
    logger.critical("IBM_PROJECT_ID is not set. Please set it in your environment variables or .env file.")
    sys.exit(1)
