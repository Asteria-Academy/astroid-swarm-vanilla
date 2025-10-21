from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

import os

load_dotenv()

class EmbedderService:
    def __init__(self):
        self._init_embedding_model()

    def _init_embedding_model(self):
        self.embedding_model = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_AZURE_EMBEDDINGS_ENDPOINT"),
            dimensions=1536
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents (batched).
        """
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.
        """
        return self.embedding_model.embed_query(query)

    @property
    def embedding_dim(self) -> int:
        return len(self.embed_query("test"))
    

if __name__ == "__main__":
    embedder = EmbedderService()

    documents = [
        "LangChain is a framework for building LLM apps.",
        "Azure OpenAI supports embedding models."
    ]
    doc_embeddings = embedder.embed_documents(documents)
    print(f"documents:\n{doc_embeddings}\nvector embeddings:\n{doc_embeddings}")

    query = "What is LangChain?"
    query_embedding = embedder.embed_query(query)
    print(f"query:\n{query}\nvector embedding:\n{query_embedding}")

    print(f"vector length: {embedder.embedding_dim}")