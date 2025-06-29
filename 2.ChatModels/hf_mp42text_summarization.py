import os
import torch
from pydub import AudioSegment
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Configuration ---
# IMPORTANT: Provide the full path to your local video file here.
# Example: "C:/Users/YourUser/Videos/my_video.mp4" or "/home/user/videos/my_video.mp4"
LOCAL_VIDEO_PATH = "D:\\Study\\MS\\GenAI\\LangChainFundamentals\\Introduction.mp4" 

# Choose a powerful ASR model from Hugging Face.
# 'base.en' is a good starting point for English. 'small', 'medium', or 'large' are more accurate.
WHISPER_MODEL_NAME = "tiny" 

# Choose a Hugging Face LLM for summarization.
# 'HuggingFaceH4/zephyr-7b-beta' is a great choice. You might need a Hugging Face token for some models.
LLM_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta" 

# --- Functions ---

def get_audio_from_local_video(video_path):
    """
    Processes a local video file to extract audio and saves it as an MP3.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file was not found at: {video_path}")
        
    audio_output_path = "extracted_audio.mp3"
    print(f"Processing local video file: {video_path}")
    
    # Use pydub to load the video and export the audio
    try:
        video = AudioSegment.from_file(video_path, format=video_path.split('.')[-1])
        video.export(audio_output_path, format="mp3")
        print(f"Audio extracted and saved to: {audio_output_path}")
    except Exception as e:
        raise Exception(f"Failed to extract audio from video. Make sure ffmpeg is installed and the video format is supported. Error: {e}")

    return audio_output_path

def transcribe_audio(audio_path, model_name=WHISPER_MODEL_NAME):
    """
    Transcribes the audio file using a Hugging Face Whisper model.
    """
    print(f"Loading Whisper model: {model_name}...")
    
    # Check for GPU and move model to it if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Download and load the Whisper model
    model = whisper.load_model(model_name.split('/')[-1], device=device)
    
    print("Transcribing audio...")
    # Perform the transcription
    result = model.transcribe(audio_path)
    
    transcription = result["text"]
    print("\n--- Transcription Complete ---")
    print(transcription[:500] + "...") # Print a snippet for verification
    
    return transcription

def generate_summary_with_langchain(transcription, llm_model_name=LLM_MODEL_NAME):
    """
    Uses a LangChain and Hugging Face LLM to generate a summary from the transcription.
    """
    print(f"\n--- Generating summary with LangChain and {llm_model_name} ---")
    
    # Check for GPU and move model to it if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the LLM using the Hugging Face transformers pipeline
    # You might need to set your Hugging Face token in your environment variables
    # or log in with `huggingface-cli login` in your terminal.
    
    # Note: Some models (like Zephyr) require specific trust_remote_code=True
    model_pipeline = pipeline(
        "text-generation",
        model=llm_model_name,
        device=0 if device.type == "cuda" else -1, # Set device for the pipeline
        max_new_tokens=512, # Adjust this value for longer summaries
        torch_dtype=torch.float16, # Use float16 for efficiency on GPU
        trust_remote_code=True,
    )
    
    # Wrap the Hugging Face pipeline in a LangChain LLM object
    llm = HuggingFacePipeline(pipeline=model_pipeline)
    
    # Define a prompt template for summarization
    prompt_template = PromptTemplate(
        input_variables=["transcription"],
        template="""You are an expert summarizer. Analyze the following video transcription and provide a concise, well-structured summary of the key points and main ideas.
        
        Transcription:
        {transcription}
        
        Summary:
        """
    )
    
    # Create the LangChain LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Run the chain to get the summary
    summary = llm_chain.invoke({"transcription": transcription})["text"]
    
    return summary

# --- Main execution flow ---
if __name__ == "__main__":
    try:
        # Step 1: Extract audio from the local video file
        audio_file = get_audio_from_local_video(LOCAL_VIDEO_PATH)
        
        # Step 2: Transcribe the audio to text using the Whisper model
        transcribed_text = transcribe_audio(audio_file)
        
        # Step 3: Use LangChain and an LLM to generate a summary from the transcription
        summary = generate_summary_with_langchain(transcribed_text)
        
        print("\n" + "="*50)
        print("Generated Summary:")
        print("="*50)
        print(summary)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # Clean up the extracted audio file
        if 'audio_file' in locals() and os.path.exists(audio_file):
            os.remove(audio_file)
            print(f"\nCleaned up the temporary audio file: {audio_file}")