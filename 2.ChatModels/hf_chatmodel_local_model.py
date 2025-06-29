from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

# --- 1. CONFIGURATION ---
# Define the model to be used
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
# Define the local directory on your D drive to save the model
# Use a raw string (r"") for Windows paths
LOCAL_MODEL_DIR = r"D:\HuggingFace_Models\zephyr-7b-beta"
# Define the directory and filename for saving the output
OUTPUT_DIRECTORY = r"D:\Study\MS\GenAI\LangChainFundamentals\output"
OUTPUT_FILENAME = "capital_of_pakistan.txt"
OUTPUT_PATH = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)

# --- 2. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not huggingface_api_token:
    print("Error: HUGGINGFACEHUB_API_TOKEN not found in .env file.")
    # Exit the script if the token is not found
    exit()
else:
    print("HUGGINGFACEHUB_API_TOKEN loaded successfully.")

# --- 3. DOWNLOAD MODEL IF NOT ALREADY PRESENT ---
# Check if the model directory exists and contains some files
if not os.path.exists(LOCAL_MODEL_DIR) or not os.listdir(LOCAL_MODEL_DIR):
    print(f"Model not found at '{LOCAL_MODEL_DIR}'. Downloading...")
    try:
        # Create the directory before downloading
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False, # This is important to get a full copy
            token=huggingface_api_token # Use the token for gated models
        )
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nError downloading model: {e}")
        # You might want to exit here if the download fails
        exit()
else:
    print(f"\nModel already exists at '{LOCAL_MODEL_DIR}'. Using local files.")

# --- 4. INITIALIZE THE LANGUAGE MODEL (LLM) ---
# Load the model from the local directory on your D drive
llm = HuggingFacePipeline.from_model_id(
    model_id=LOCAL_MODEL_DIR, # Pass the local directory path as the model_id
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 512,
        "do_sample": True
    }
)

# --- 5. INITIALIZE THE CHAT MODEL ---
model = ChatHuggingFace(llm=llm)

# --- 6. INVOKE THE MODEL ---
print("\n--- Invoking the model ---")
try:
    result = model.invoke("What is the capital of Pakistan?")
    response_content = result.content

    # --- 7. SAVE THE OUTPUT TO A FILE ---
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(response_content)
    print(f"\nResponse successfully saved to: {OUTPUT_PATH}")
    
    # --- 8. PRINT THE OUTPUT TO THE CONSOLE ---
    print("\n--- Model Response ---")
    print(response_content)

except Exception as e:
    print(f"\nAn error occurred during model inference or file saving: {e}")