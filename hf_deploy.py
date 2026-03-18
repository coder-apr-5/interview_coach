from huggingface_hub import HfApi
from dotenv import dotenv_values
import os

def deploy_to_hf():
    api = HfApi()
    
    # 1. Get identity
    user = api.whoami()["name"]
    repo_id = f"{user}/ai-interview-coach"
    print(f"📡 Verified Identity: {user}")
    
    # 2. Create hugging face space (if not exists)
    print(f"🚀 Initializing Hugging Face Space: https://huggingface.co/spaces/{repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
    
    # 3. Apply Secrets
    env_vars = dotenv_values(".env")
    if "GROQ_API_KEY" in env_vars:
        print("🔐 Injecting GROQ_API_KEY securely into Space instance...")
        api.add_space_secret(repo_id, "GROQ_API_KEY", env_vars["GROQ_API_KEY"])
        
    # 4. Upload Files safely utilizing .huggingfaceignore
    print("☁️ Uploading required files (bypassing heavy venv binaries)...")
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["venv/*", ".git/*", "__pycache__/*", ".env"] # Extra safety net in case ignore file fails
    )
    
    print(f"✅ DEPLOYMENT COMPLETE! Your permanent app URL is:\nhttps://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    deploy_to_hf()
