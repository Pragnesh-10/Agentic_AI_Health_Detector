import os
from ibm_watsonx_ai.foundation_models import get_embedding_model_specs
from ibm_watsonx_ai import Credentials
from dotenv import load_dotenv

load_dotenv()

credentials = Credentials(
    url=os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
    api_key=os.getenv("IBM_API_KEY")
)

models = get_embedding_model_specs(credentials.url)
print(models)
