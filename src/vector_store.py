import logging
from urllib.parse import unquote
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from minio import Minio
from tika import parser
import yaml

class VectorStore:
    """
    A class to manage a vector store for document retrieval. It integrates with Minio for document storage,
    Tika for document parsing, and FAISS for vector-based similarity search.
    """

    def __init__(self, config_path):
        """
        Initializes the VectorStore with configurations from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        try:
            # Initialize logger for tracking events and errors
            self.logger = logging.getLogger("uvicorn")
        except Exception as e:
            logging.error(f"Failed to get self.logger: {e}")
        
        self.logger.info('Initializing VectorStore...')
        
        # Load configuration from the YAML file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Minio client for document storage
        self.logger.info('Initializing Minio Client...')
        self.minio_client = Minio(
            self.config['minio']['endpoint'],
            access_key=self.config['minio']['access_key'],
            secret_key=self.config['minio']['secret_key'],
            secure=self.config['minio']['secure']
        )
        self.logger.info('Minio Client initialized successfully.')
        
        # Initialize embedding model for text vectorization
        self.logger.info('Initializing Embedding model...')
        self.embeddings = OllamaEmbeddings(
            model=self.config['embedding_model']['model_name']
        )
        self.logger.info('Embedding model initialized successfully.')
        
        # Initialize text splitter for chunking documents
        self.logger.info('Initializing CharacterTextSplitter...')
        self.chunker = CharacterTextSplitter(
            chunk_size=self.config["text_chunker"]["chunk_size"],
            chunk_overlap=self.config["text_chunker"]["chunk_overlap"]
        )
        self.logger.info('CharacterTextSplitter initialized successfully.')
        
        # Initialize the vector store and index
        self.vector_store = None
        self.initialize_index()

    def process_document(self, file_data):
        """
        Processes a document using Apache Tika to extract its content.

        Args:
            file_data (bytes): The raw file data to be parsed.

        Returns:
            str: The extracted text content of the document.
        """
        parsed = parser.from_buffer(file_data)
        return parsed["content"]

    def chunk_text(self, text):
        """
        Splits a text into smaller chunks using the configured text splitter.

        Args:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        return self.chunker.split_text(text)

    def initialize_index(self):
        """
        Initializes the FAISS index by processing documents from the Minio bucket.
        Each document is parsed, chunked, and added to the vector store.
        """
        self.logger.info("Initializing index from scratch...")
        docs_bucket = self.config['minio']['docs_bucket']
        documents = []
        
        # Iterate over all objects in the Minio bucket
        for obj in self.minio_client.list_objects(docs_bucket):
            try:
                self.logger.info(f"Processing document: {unquote(obj.object_name)}")
                
                # Retrieve the document from Minio
                obj_response = self.minio_client.get_object(docs_bucket, unquote(obj.object_name))
                
                # Parse the document content using Tika
                self.logger.info(f"TIKA: parsing document: {unquote(obj.object_name)}")
                content = self.process_document(obj_response)
                self.logger.info(f"TIKA: done parsing document: {unquote(obj.object_name)}")
                
                # Split the document content into chunks
                self.logger.info(f"Chunking document: {unquote(obj.object_name)}")
                chunks = self.chunk_text(content)
                
                # Create Document objects for each chunk
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata={"source": unquote(obj.object_name)}))
            except Exception as e:
                self.logger.error(f"Failed to process document {unquote(obj.object_name)}: {e}")
        
        # Build the FAISS index from the processed documents
        self.logger.info("Building FAISS index...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.logger.info("FAISS index initialization complete.")
        
        # Initialize the retriever for similarity search
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 5})

    def add_document(self, file_data, object_name):
        """
        Adds a new document to the vector store.

        Args:
            file_data (bytes): The raw file data to be added.
            object_name (str): The name of the object in the Minio bucket.
        """
        # Process the document and split it into chunks
        content = self.process_document(file_data)
        chunks = self.chunk_text(content)
        
        # Create Document objects for each chunk and add them to the vector store
        documents = [Document(page_content=chunk, metadata={"source": object_name}) for chunk in chunks]
        self.vector_store.add_documents(documents)

    def search(self, query, k=5):
        """
        Performs a similarity search on the vector store.

        Args:
            query (str): The query string to search for.
            k (int): The number of results to return (default: 5).

        Returns:
            list: A list of documents that are most similar to the query.
        """
        return self.vector_store.similarity_search(query, k=k)