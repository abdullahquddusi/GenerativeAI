from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# Corrected call to from_model_id
llm = HuggingFacePipeline.from_model_id(
    "HuggingFaceH4/zephyr-7b-beta",  # This is the 'model_id' positional argument
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100
    }
)

model = ChatHuggingFace(llm=llm)
    
# Invoke the model with a clear chat message
result = model.invoke("What is the capital of Pakistan?")
    
print(result.content)