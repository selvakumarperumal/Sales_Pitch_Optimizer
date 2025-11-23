from typing import List, Dict, Any
from dataclasses import dataclass

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder

# Qdrant imports
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
# Note: Switched to EmbeddingRetriever because Ollama only produces dense embeddings. 
# HybridRetriever requires a Sparse Embedder (like FastEmbed or SPLADE) as well.
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever


@dataclass
class CustomerProfile:
    """Customer information from CRM"""
    name: str
    company: str
    industry: str
    pain_points: List[str]
    budget_range: str
    previous_interactions: List[str]

def prepare_sales_document() -> List[Document]:
    """
    Create Mock Sales Documents
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
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        # Create a Qdrant document store instance
        # Note: We use "recreate_index=True" for this demo to ensure a clean start each run.
        self.document_store = QdrantDocumentStore(
            path="./qdrant_storage_local",   # Local folder for storage
            index="Document",                
            embedding_dim=768,               # Adjusted to 768 for nomic-embed-text (check your model dims!)
            recreate_index=True,             
            hnsw_config={                    
                "m": 16,                     
                "ef_construct": 64           
            }
        )

        print(f"Initializing Embedder with model: {embedding_model}...")
        self.document_embedder = OllamaDocumentEmbedder(
            model=embedding_model,
            url="http://localhost:11434",
        )
        
        print("Indexing documents...")
        docs_with_embeddings = self.document_embedder.run(
            documents=prepare_sales_document()
        )["documents"]

        self.document_store.write_documents(docs_with_embeddings)
        print(f"Indexed {len(docs_with_embeddings)} documents with embeddings.")

        self.pipeline = self._build_pipeline(embedding_model)

    def _build_pipeline(self, embedding_model: str) -> Pipeline:

        text_embedder = OllamaTextEmbedder(
            model=embedding_model,
            url="http://localhost:11434",
        )

        # Using Dense Retriever (Standard for Ollama)
        retriever = QdrantEmbeddingRetriever(
            document_store=self.document_store,
            top_k=3
        )

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

        prompt_builder = PromptBuilder(template=template)

        generator = OllamaGenerator(
            model="mistral",
            url="http://localhost:11434",
            generation_kwargs={
                "num_predict": 500,
                "temperature": 0.7
            }
        )

        pipeline = Pipeline()
        pipeline.add_component(name="text_embedder", instance=text_embedder)
        pipeline.add_component(name="retriever", instance=retriever)
        pipeline.add_component(name="prompt_builder", instance=prompt_builder)
        pipeline.add_component(name="generator", instance=generator)

        # Connect the components
        # 1. Embed query -> Retrieve Docs
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        
        # 2. Retrieved Docs -> Prompt Builder
        pipeline.connect("retriever.documents", "prompt_builder.retrieved_documents")
        
        # 3. Prompt -> Generator
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        
        return pipeline

    def generate_pitch(self, customer: CustomerProfile, task: str) -> str:
        """
        Executes the pipeline to generate a pitch.
        """
        print("Running pipeline...")
        result = self.pipeline.run({
            "text_embedder": {"text": task}, # We use the task description as the search query
            "prompt_builder": {
                "customer_name": customer.name,
                "company": customer.company,
                "industry": customer.industry,
                "pain_points": ", ".join(customer.pain_points),
                "budget_range": customer.budget_range,
                "previous_interactions": "; ".join(customer.previous_interactions),
                "task_description": task,
                # "retrieved_documents" is populated by the retriever connection
            }
        })
        
        return result["generator"]["replies"][0]

if __name__ == "__main__":
    # Initialize Optimizer
    # Ensure you have 'nomic-embed-text' pulled in Ollama for embeddings
    # and 'mistral' pulled for generation.
    optimizer = SalesPitchOptimizer(embedding_model="nomic-embed-text")

    # Define a Mock Customer
    customer = CustomerProfile(
        name="John Doe",
        company="Tech Innovators",
        industry="Software Development",
        pain_points=["Inefficient data analysis", "Slow customer response times"],
        budget_range="$50k - $100k",
        previous_interactions=["Inquired about analytics tools", "Downloaded whitepaper"]
    )

    task = "Propose our analytics platform to solve their efficiency issues."

    # Generate Pitch
    pitch = optimizer.generate_pitch(customer, task)
    
    print("\n" + "="*50)
    print("GENERATED SALES PITCH")
    print("="*50 + "\n")
    print(pitch)