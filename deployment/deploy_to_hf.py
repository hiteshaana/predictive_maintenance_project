"""
Predictive Maintenance - Hugging Face Space Deployment

This script uploads the complete deployment application
to a Hugging Face Docker Space.

Required environment variable:
    HF_TOKEN

Optional environment variables:
    HF_SPACE_ID
"""

import os
from pathlib import Path

from huggingface_hub import HfApi


# ============================================================
# Configuration
# ============================================================

# Hugging Face Space repository
SPACE_ID = os.getenv(
    "HF_SPACE_ID",
    "hiteshsharma/predictive-maintenance"
)

# Hugging Face authentication token
HF_TOKEN = os.getenv("HF_TOKEN")

# Directory containing deployment files
DEPLOYMENT_DIR = Path(__file__).resolve().parent


# ============================================================
# Validate configuration
# ============================================================

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. "
        "Please configure your Hugging Face access token."
    )


required_files = [
    "app.py",
    "Dockerfile",
    "requirements.txt",
]

missing_files = [
    file_name
    for file_name in required_files
    if not (DEPLOYMENT_DIR / file_name).exists()
]

if missing_files:
    raise FileNotFoundError(
        "The following required deployment files are missing: "
        + ", ".join(missing_files)
    )


# ============================================================
# Initialize Hugging Face API
# ============================================================

api = HfApi(token=HF_TOKEN)


# ============================================================
# Create / verify Hugging Face Space
# ============================================================

print("=" * 70)
print("Predictive Maintenance - Hugging Face Deployment")
print("=" * 70)

print(f"Space ID: {SPACE_ID}")
print(f"Deployment directory: {DEPLOYMENT_DIR}")

print("\nCreating/verifying Hugging Face Space...")

api.create_repo(
    repo_id=SPACE_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)

print("Hugging Face Space is ready.")


# ============================================================
# Upload deployment files
# ============================================================

print("\nUploading deployment files...")

api.upload_folder(
    repo_id=SPACE_ID,
    repo_type="space",
    folder_path=str(DEPLOYMENT_DIR),
    allow_patterns=[
        "app.py",
        "Dockerfile",
        "requirements.txt",
    ],
    commit_message="Deploy predictive maintenance Streamlit application",
)

print("\nDeployment files uploaded successfully.")


# ============================================================
# Deployment summary
# ============================================================

space_url = f"https://huggingface.co/spaces/{SPACE_ID}"

print("\n" + "=" * 70)
print("DEPLOYMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Space URL: {space_url}")
print("\nFiles deployed:")
print("  - app.py")
print("  - Dockerfile")
print("  - requirements.txt")

print("\nHugging Face will now build and start the Docker Space.")
print("Please open the Space URL and verify the application.")
