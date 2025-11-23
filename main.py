from typing import List, Dict, Any
from dataclasses import dataclass

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.utils import Secret
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder

# Qdrant imports for vector database functionality
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
# Note: Switched to EmbeddingRetriever because Ollama only produces dense embeddings. 
# HybridRetriever requires a Sparse Embedder (like FastEmbed or SPLADE) as well.
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder, FastembedSparseDocumentEmbedder
from dotenv import load_dotenv
import os

load_dotenv("./.env")  # Load environment variables from .env file

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")


@dataclass
class CustomerProfile:
    """
    Data class to store customer information from CRM system.
    This helps organize customer data in a structured way.
    """
    name: str                           # Customer's full name
    company: str                        # Customer's company name
    industry: str                       # Industry sector (e.g., Software, Healthcare)
    pain_points: List[str]              # List of customer's business challenges
    budget_range: str                   # Available budget for purchase
    previous_interactions: List[str]    # History of past communications

def prepare_sales_document() -> List[Document]:
    """
    Create Mock Sales Documents containing product/service information.
    
    In a real application, these would be loaded from:
    - Product documentation files
    - Marketing materials database
    - Case studies repository
    
    Returns:
        List[Document]: Collection of Haystack Document objects
    """
    product_docs = [
        Document(content="Our AI-powered analytics platform helps businesses gain insights from their data quickly and efficiently."),
        Document(content="We offer a cloud-based CRM solution that streamlines customer management and improves sales team productivity."),
        Document(content="Our cybersecurity services protect your business from threats with advanced monitoring and response capabilities."),
        Document(content="We provide scalable cloud infrastructure solutions that adapt to your business needs and ensure high availability."),
        Document(content="DataFlow Pro automation reduces manual data entry by 85% and integrates with Salesforce."),
    ]
    return product_docs

class SalesPitchOptimizer:
    """
    Main class that orchestrates the sales pitch generation process.
    Uses RAG (Retrieval Augmented Generation) to create personalized pitches.
    """
    
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        """
        Initialize the optimizer with document store and embeddings.
        
        Args:
            embedding_model: Name of the Ollama model for generating embeddings
        """
        # Create a Qdrant document store instance (vector database)
        # Qdrant stores document embeddings and enables semantic search

        if QDRANT_URL and QDRANT_API_KEY:
            self.document_store = QdrantDocumentStore(
                url=QDRANT_URL,
                api_key=Secret.from_token(QDRANT_API_KEY),  # Wrap API key in Secret object
                index="Document",                # Name of the index/collection
                embedding_dim=768,               # Dimension of dense embeddings (nomic-embed-text uses 768)
                use_sparse_embeddings=True,      # Enable sparse embeddings for hybrid retrieval (better accuracy)
                recreate_index=True,             # Recreate index on each run (clean slate for demo)
                hnsw_config={                    # HNSW algorithm config for fast approximate search
                    "m": 16,                     # Number of connections per node (higher = more accurate but slower)
                    "ef_construct": 64           # Size of dynamic candidate list (higher = better index quality)
                }
            )
        else:
            self.document_store = QdrantDocumentStore(
                path="./qdrant_storage_local",   # Local folder to persist vector data
                index="Document",                # Name of the index/collection
                embedding_dim=768,               # Dimension of dense embeddings (nomic-embed-text uses 768)
                use_sparse_embeddings=True,      # Enable sparse embeddings for hybrid retrieval (better accuracy)
                recreate_index=True,             # Recreate index on each run (clean slate for demo)
                hnsw_config={                    # HNSW algorithm config for fast approximate search
                    "m": 16,                     # Number of connections per node (higher = more accurate but slower)
                    "ef_construct": 64           # Size of dynamic candidate list (higher = better index quality)
                }
            )

        print(f"Initializing Embedder with model: {embedding_model}...")

        # Dense embedder: Converts documents to dense vectors (continuous numerical representation)
        # Used for semantic similarity search
        self.dense_embedder = OllamaDocumentEmbedder(
            model=embedding_model,
            url="http://localhost:11434",  # Local Ollama server
        )
        
        # Sparse embedder: Converts documents to sparse vectors (keyword-based representation)
        # Complements dense embeddings by capturing exact term matches
        self.sparse_embedder = FastembedSparseDocumentEmbedder(model="prithivida/Splade_PP_en_v1")
        self.sparse_embedder.warm_up()  # Load model into memory for faster processing

        print("Indexing documents...")
        
        # Step 1: Generate dense embeddings for all documents
        dense_result = self.dense_embedder.run(documents=prepare_sales_document())
        docs_with_dense = dense_result["documents"]

        # Step 2: Add sparse embeddings to the same documents
        sparse_result = self.sparse_embedder.run(documents=docs_with_dense)
        docs_with_both = sparse_result["documents"]  # Documents now have both embedding types

        # Step 3: Store documents with embeddings in Qdrant
        self.document_store.write_documents(docs_with_both)
        print(f"Indexed {len(docs_with_both)} documents with embeddings.")

        # Build the RAG pipeline
        self.pipeline = self._build_pipeline(embedding_model)

    def _build_pipeline(self, embedding_model: str) -> Pipeline:
        """
        Construct the Haystack pipeline for RAG-based pitch generation.
        
        Pipeline flow:
        1. Embed user query (dense + sparse)
        2. Retrieve relevant documents from vector store
        3. Build prompt with customer info + retrieved docs
        4. Generate personalized pitch using LLM
        
        Args:
            embedding_model: Model name for query embeddings
            
        Returns:
            Pipeline: Configured Haystack pipeline
        """

        # Text embedder: Converts query text to dense vector for semantic search
        text_embedder = OllamaTextEmbedder(
            model=embedding_model,
            url="http://localhost:11434",
        )

        # Sparse text embedder: Converts query to sparse vector for keyword matching
        sparse_text_embedder = FastembedSparseTextEmbedder(model="prithivida/Splade_PP_en_v1")
        sparse_text_embedder.warm_up()

        # Hybrid retriever: Combines dense (semantic) + sparse (keyword) search
        # Returns most relevant documents based on both similarity types
        hybrid_retriever = QdrantHybridRetriever(
            document_store=self.document_store,
            top_k=3,  # Return top 3 most relevant documents
        )

        # Prompt template: Structures the input for the LLM
        # Uses Jinja2 syntax for variable substitution
        template = """
        You are a sales pitch generator. Based on the customer profile and product information, create a compelling sales pitch.

        Customer Context:
        Name: {{customer_name}}
        Company: {{company}}
        Industry: {{industry}}
        Pain Points: {{pain_points}}
        Budget Range: {{budget_range}}
        Previous Interactions: {{previous_interactions}}

        Relevant Information:
        {% for doc in retrieved_documents %}
        - {{ doc.content }}
        {% endfor %}

        Task: {{ task_description }}

        Generate a compelling, personalized sales pitch that:
        - Addresses the customer's pain points
        - Highlights relevant product features and case studies
        - Positions against competitors
        - Includes a clear call to action
        - Maintains a professional and persuasive tone
        """

        # Prompt builder: Fills template with actual data
        prompt_builder = PromptBuilder(template=template)

        # Generator: LLM that creates the final sales pitch
        generator = OllamaGenerator(
            model="mistral",  # Using Mistral LLM for generation
            url="http://localhost:11434",
            generation_kwargs={
                "num_predict": 500,      # Maximum tokens to generate
                "temperature": 0.7       # Creativity level (0=deterministic, 1=creative)
            }
        )

        # Create pipeline and add all components
        pipeline = Pipeline()
        pipeline.add_component(name="text_embedder", instance=text_embedder)
        pipeline.add_component(name="sparse_text_embedder", instance=sparse_text_embedder)
        pipeline.add_component(name="hybrid_retriever", instance=hybrid_retriever)
        pipeline.add_component(name="prompt_builder", instance=prompt_builder)
        pipeline.add_component(name="generator", instance=generator)

        # Connect components to create data flow
        # Step 1: Query embeddings flow into retriever
        pipeline.connect("text_embedder.embedding", "hybrid_retriever.query_embedding")
        pipeline.connect("sparse_text_embedder.sparse_embedding", "hybrid_retriever.query_sparse_embedding")
        
        # Step 2: Retrieved documents flow into prompt builder
        pipeline.connect("hybrid_retriever.documents", "prompt_builder.retrieved_documents")
        
        # Step 3: Constructed prompt flows into LLM generator
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        
        return pipeline

    def generate_pitch(self, customer: CustomerProfile, task: str) -> str:
        """
        Execute the complete pipeline to generate a personalized sales pitch.
        
        Args:
            customer: CustomerProfile object with customer details
            task: Description of what to pitch (used as search query)
            
        Returns:
            str: Generated sales pitch text
        """
        print("Running pipeline...")
        
        # Execute pipeline with all inputs
        result = self.pipeline.run({
            # Query embeddings: Task description is used to find relevant products
            "text_embedder": {"text": task},
            "sparse_text_embedder": {"text": task},
            
            # Prompt variables: Customer information for personalization
            "prompt_builder": {
                "customer_name": customer.name,
                "company": customer.company,
                "industry": customer.industry,
                "pain_points": ", ".join(customer.pain_points),      # Convert list to comma-separated string
                "budget_range": customer.budget_range,
                "previous_interactions": "; ".join(customer.previous_interactions),  # Convert list to semicolon-separated
                "task_description": task,
                # Note: "retrieved_documents" is automatically populated by the retriever connection
            }
        })
        
        # Extract generated text from pipeline output
        return result["generator"]["replies"][0]

if __name__ == "__main__":
    """
    Main execution block - demonstrates the system with a sample customer.
    
    Prerequisites:
    1. Ollama must be running locally (ollama serve)
    2. Required models must be pulled:
       - ollama pull nomic-embed-text (for embeddings)
       - ollama pull mistral (for text generation)
    """
    
    # Initialize the optimizer system
    optimizer = SalesPitchOptimizer(embedding_model="nomic-embed-text")

    # Create a sample customer profile
    customer = CustomerProfile(
        name="John Doe",
        company="Tech Innovators",
        industry="Software Development",
        pain_points=["Inefficient data analysis", "Slow customer response times"],
        budget_range="$50k - $100k",
        previous_interactions=["Inquired about analytics tools", "Downloaded whitepaper"]
    )

    # Define what to pitch (this will be used to search for relevant products)
    task = "Propose our analytics platform to solve their efficiency issues."

    # Generate the personalized sales pitch
    pitch = optimizer.generate_pitch(customer, task)
    
    # Display the result
    print("\n" + "="*50)
    print("GENERATED SALES PITCH")
    print("="*50 + "\n")
    print(pitch)