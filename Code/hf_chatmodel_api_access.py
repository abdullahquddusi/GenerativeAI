from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

# Check if the token is loaded
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not huggingface_api_token:
    print("Error: HUGGINGFACEHUB_API_TOKEN not found in .env file.")
else:
    print("HUGGINGFACEHUB_API_TOKEN loaded successfully.")
    
    # Use a model known to work with the chat API on Hugging Face
    llm1 = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta", # A good alternative for chat
        task="text-generation",
        # Set max_new_tokens to avoid timeouts for longer responses
        max_new_tokens=512, 
        do_sample=True, # Recommended for text generation
        temperature=0.7 # Recommended for text generation
    )
    
    # Use ChatHuggingFace with the LLM
    model = ChatHuggingFace(llm=llm1)
    
    # Invoke the model with a clear chat message
    result = model.invoke("How many States USA have?")
    
    print(result.content)