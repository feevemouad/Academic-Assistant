# Import necessary libraries and modules
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from urllib.parse import unquote
import uvicorn
import os
import yaml
import io

# Import custom modules and classes
from src.chat_model import ChatModel
from src.vector_store import VectorStore
from src.models.chat_model import ChatRequest

# Load configuration from a YAML file
config_path = 'config.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize the vector store with the configuration
vector_store = VectorStore(config_path)

# Set up LangSmith environment variables for tracing (if configured)
try:
    os.environ["LANGCHAIN_TRACING_V2"] = config["langsmith"]["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_ENDPOINT"] = config["langsmith"]["LANGCHAIN_ENDPOINT"]
    os.environ["LANGCHAIN_API_KEY"] = config["langsmith"]["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_PROJECT"] = config["langsmith"]["LANGCHAIN_PROJECT"]
except Exception as e:
    vector_store.logger.warning(f"LangSmith environment variables not configured: {str(e)}")
    vector_store.logger.info(f"LangSmith environment variables not configured: {str(e)}")

# Create a FastAPI application instance
app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Endpoint to handle chat requests. It takes a user input and conversation history,
    generates a response using the chat model, and returns the answer along with source documents.
    """
    try:
        # Log the incoming chat request for debugging purposes
        vector_store.logger.info(f"Received chat request: {request.dict()}")
        
        # Initialize the chat model with the provided LLM configuration and vector store
        chat_model = ChatModel(vector_store, request.llm_model)

        # Generate the response using the chat model
        answer, source_documents = chat_model.generate_response(
            request.user_input,
            request.conversation_history
        )

        # Extract metadata (e.g., source file names) from the source documents
        sources = [doc.metadata['source'] for doc in source_documents]

        # Return the response as JSON, including the generated answer and source documents
        return JSONResponse(content={"answer": answer, "source_documents": sources})

    except Exception as e:
        # Handle any errors that occur during processing and return a 500 status code with the error message
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook")
async def webhook(request: Request):
    """
    Webhook endpoint to handle S3 bucket events (e.g., file uploads or deletions).
    When a file is uploaded or deleted in the S3 bucket, this webhook updates the vector store index accordingly.
    """
    try:
        # Parse the incoming JSON payload from the webhook request
        content = await request.json()
        
        # Iterate through each record in the webhook payload
        for record in content['Records']:
            try:
                # Extract the event name (e.g., "ObjectCreated" or "ObjectRemoved")
                event_name = record['eventName']

                # Extract the bucket name where the event occurred
                bucket_name = record['s3']['bucket']['name']

                # Extract the object name (file name) that triggered the event
                object_name = record['s3']['object']['key']
                # Log the object name for debugging purposes
                vector_store.logger.info(f"Processing object: {object_name}")

                # Check if the event is related to an object being created (e.g., file upload)
                if "ObjectCreated" in event_name:
                    # Fetch the uploaded file from the S3 bucket
                    obj_response = vector_store.minio_client.get_object(bucket_name, unquote(object_name))

                    # Convert the file content into a file-like object (BytesIO) for processing
                    file_like_data = io.BytesIO(obj_response.read())

                    # Add the document to the vector store index
                    vector_store.add_document(file_like_data, object_name)

                # Check if the event is related to an object being removed (e.g., file deletion)
                elif "ObjectRemoved" in event_name:
                    # Reinitialize the vector store index to reflect the removal of the document
                    vector_store.initialize_index()
            except Exception as e:
                vector_store.logger.error(f"Error processing record: {record}, error: {str(e)}")

        # Return a success response indicating the index was updated successfully
        return JSONResponse(content={"message": "Index updated successfully."})
    except Exception as e:
        vector_store.logger.error(f"Error processing webhook request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)