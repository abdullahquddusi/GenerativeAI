import streamlit as st
import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_core.messages import HumanMessage # Import for chat messages

# --- Load Environment Variables ---
# This will load variables from a .env file if it exists.
# Make sure to have HUGGINGFACEHUB_API_TOKEN in your .env file
load_dotenv()

# --- Configuration for Question Answering Model ---
# You can choose a different question-answering model if you prefer.
# Some popular options include:
# - 'distilbert-base-uncased-distilled-squad' (smaller, faster)
# - 'deepset/roberta-base-squad2' (larger, often more accurate)
# - 'bert-large-uncased-whole-word-masking-finetuned-squad'
QA_MODEL_NAME = "distilbert-base-uncased-distilled-squad"

# --- Initialize the Question Answering Pipeline ---
# Using st.cache_resource to cache the model loading, so it only loads once.
@st.cache_resource
def load_qa_pipeline(model_name: str):
    """
    Loads the Hugging Face question-answering pipeline.
    Caches the resource to prevent reloading on every rerun.
    """
    try:
        qa_pipe = pipeline("question-answering", model=model_name)
        st.success(f"Successfully loaded Hugging Face QA model: {model_name}")
        return qa_pipe
    except Exception as e:
        st.error(f"Error loading QA model {model_name}: {e}")
        st.info("Please ensure you have an active internet connection to download the model.")
        st.info("You might also need to install 'transformers': pip install transformers")
        return None

qa_pipeline = load_qa_pipeline(QA_MODEL_NAME)

# --- Function to Answer Marketing Questions ---
def answer_marketing_question(qa_pipe, question: str, context: str) -> dict:
    """
    Uses the loaded Hugging Face model to answer a question based on a given context.

    Args:
        qa_pipe: The initialized question-answering pipeline.
        question (str): The marketing-related question to answer.
        context (str): The text context from which to extract the answer.

    Returns:
        dict: A dictionary containing the answer, score, start, and end positions.
              Example: {'answer': '...', 'score': 0.99, 'start': 10, 'end': 20}
    """
    if not qa_pipe:
        return {"answer": "QA model not loaded.", "score": 0.0, "start": -1, "end": -1}
    if not question or not context:
        return {"answer": "Please provide both a question and a context.", "score": 0.0, "start": -1, "end": -1}

    try:
        result = qa_pipe(question=question, context=context)
        return result
    except Exception as e:
        st.error(f"An error occurred during question answering: {e}")
        return {"answer": "Could not process the question.", "score": 0.0, "start": -1, "end": -1}

# --- Configuration for LangChain Chat Model ---
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

@st.cache_resource
def load_chat_model(token: str):
    """
    Loads the LangChain Hugging Face chat model.
    Caches the resource to prevent reloading on every rerun.
    """
    if not token:
        st.warning("HUGGINGFACEHUB_API_TOKEN not found. General chat functionality will be disabled.")
        st.info("Please add HUGGINGFACEHUB_API_TOKEN='your_token_here' to a .env file in the same directory.")
        return None
    try:
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta", # A good alternative for chat
            task="text-generation",
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            huggingfacehub_api_token=token # Pass token directly
        )
        chat_model = ChatHuggingFace(llm=llm)
        st.success("Successfully loaded LangChain Chat model.")
        return chat_model
    except Exception as e:
        st.error(f"Error loading LangChain Chat model: {e}")
        st.info("Ensure the API token is correct and the model is accessible.")
        return None

chat_model = load_chat_model(huggingface_api_token)

# --- Streamlit Application Layout ---
st.set_page_config(page_title="AI Marketing Assistant", layout="wide")

st.title("🤖 AI Marketing Assistant")
st.markdown("""
This application uses Hugging Face models to help you with marketing-related questions.
It features a **Question Answering** tool based on a provided context and a **General Chat**
for broader queries.
""")

# --- Tabs for different functionalities ---
tab1, tab2 = st.tabs(["Question Answering", "General Chat"])

with tab1:
    st.header("❓ Marketing Question Answering")
    st.markdown("Provide a marketing text, and ask a specific question to get an answer extracted from it.")

    marketing_context_input = st.text_area(
        "Enter Marketing Context Here:",
        height=300,
        placeholder="e.g., Content marketing is a strategic approach focused on creating and distributing valuable content..."
    )

    question_input = st.text_input(
        "Your Question:",
        placeholder="e.g., What is content marketing?"
    )

    if st.button("Get Answer", use_container_width=True):
        if qa_pipeline:
            if marketing_context_input and question_input:
                with st.spinner("Finding the answer..."):
                    qa_result = answer_marketing_question(qa_pipeline, question_input, marketing_context_input)
                    st.subheader("Answer:")
                    if qa_result['answer']:
                        st.success(qa_result['answer'])
                        st.write(f"Confidence Score: {qa_result['score']:.2f}")
                    else:
                        st.warning("Could not find a direct answer in the provided context.")
            else:
                st.warning("Please enter both context and a question.")
        else:
            st.error("Question Answering model failed to load. Please check your internet connection or `transformers` installation.")

with tab2:
    st.header("💬 General Chat")
    st.markdown("Ask general questions using a LangChain Hugging Face chat model.")

    if chat_model:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What can I help you with?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                try:
                    # Invoke the LangChain model
                    response = chat_model.invoke(HumanMessage(content=prompt))
                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(response.content)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                except Exception as e:
                    st.error(f"Error during chat interaction: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})
    else:
        st.warning("General chat functionality is disabled because the LangChain model could not be loaded.")
        st.info("Please ensure you have set the `HUGGINGFACEHUB_API_TOKEN` environment variable in a `.env` file.")

