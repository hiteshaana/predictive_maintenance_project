# ===========CODE=============================================================
# Models Registration on Hugging Face
# ============================================================================

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Define the repository ID
repo_id = "hiteshsharma/predictive-maintenance-model"

# Retrieve the Hugging Face token from environment variables
token = os.getenv("HF_TOKEN")

# Initialize HfApi client
api = HfApi(token=token)

# ----------------------------------------------------------
# Create repository if it doesn't exist
# ----------------------------------------------------------
try:
    api.repo_info(
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating repository '{repo_id}'...")
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        token=token
    )

# ----------------------------------------------------------
# Upload model folder
# ----------------------------------------------------------

api.upload_folder(
    folder_path="predictive_maintenance_project/model",
    repo_id=repo_id,
    repo_type="model",
    token=token
)

print("✅ Model uploaded successfully.")
