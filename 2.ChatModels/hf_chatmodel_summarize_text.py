import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()
st.header("Research Tool")

# Check if the token is loaded
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not huggingface_api_token:
    st.error("Error: HUGGINGFACEHUB_API_TOKEN not found in .env file.")
else:
    # Use a model known to work with the chat API on Hugging Face
    llm1 = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )
    
    # Use ChatHuggingFace with the LLM
    model = ChatHuggingFace(llm=llm1)
    
    # Use st.session_state to store the user input and avoid re-running the model on every interaction
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
        
    user_input = st.text_input("Enter your prompt", key="user_input_key")

    # Correct way to use a button:
    if st.button("Generate Response"):
        if user_input: # Check if the user has entered a prompt
            with st.spinner("Generating response..."): # Show a loading spinner
                result = model.invoke(user_input)
            st.write(result.content)
        else:
            st.warning("Please enter a prompt before clicking the button.")