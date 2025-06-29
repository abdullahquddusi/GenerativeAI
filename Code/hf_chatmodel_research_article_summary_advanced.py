import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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
    
    paper_input = st.selectbox("Select Research Paper Name",["Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers","GPT-3: Language Models are Few-Shot Learners"])

    style_input = st.selectbox("Select Explaination Style",["Begginer-Friendly","Technical","Code-Oriented","Mathematical"])

    length_input = st.selectbox("Select Explaination Length",["Short (1-2 paragraphs)","Medium (3-5 paragraphs)","Long (Detailed Explaination)"])
    
    #template
    template = PromptTemplate(
        template="""
    Please summarize the research paper titled "{paper_input}" with the following specificaions:
    Explaination Style : {style_input}
    Explaination Length : {length_input}
    1. Mathematical Details:
        -Include relevant mathematical equations if present in the paper.
        -Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    2. Analogies:
        -Use relatable analogies to simplify complex ideas.
    If certain information is not available in the paper, respond with "Insufficient Information available" instead of guessing.
    Ensure the summary is clear, accurate, and aligned with the provided style and length.
        """,
        input_variables=['paper_input','style_input','length_input']
    )
    
    #fill the placeholders
    prompt = template.invoke({
        'paper_input' : paper_input,
        'style_input' : style_input,
        'length_input' : length_input
    })

    # Use ChatHuggingFace with the LLM
    model = ChatHuggingFace(llm=llm1)
    
    # Use st.session_state to store the user input and avoid re-running the model upon every interaction
    

    # Correct way to use a button:
    if st.button("Generate Response"):
        with st.spinner("Generating response..."): # Show a loading spinner
                result = model.invoke(prompt)
                st.write(result.content)
       