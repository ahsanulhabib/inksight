import networkx as nx
from typing import Optional
from lightrag.core import Generator, Retriever
from lightrag.components.retriever import FAISSRetriever
from lightrag.components.generator import OllamaGenerator
from lightrag.core import Component


class GraphRAG(Component):
    def __init__(self, graph: Optional[nx.Graph] = None):
        super().__init__()
        self.graph = graph
        self.retriever: Optional[Retriever] = None
        self.generator: Optional[Generator] = None
        if self.graph:
            self.build()

    def build(self):
        """Builds the retriever and generator from the graph."""
        if not self.graph or self.graph.number_of_nodes() == 0:
            print("Graph is empty. RAG pipeline not built.")
            return

        # 1. Create text chunks from the graph to be indexed
        documents = []
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "N/A")
            neighbors = list(self.graph.neighbors(node))
            doc = (
                f"Node: {node} (Type: {node_type}). Connections: {', '.join(neighbors)}"
            )
            documents.append(doc)

        # 2. Build the FAISS Retriever
        self.retriever = FAISSRetriever(documents=documents)

        # 3. Build the Ollama Generator
        self.generator = OllamaGenerator(
            model="qwen2.5-vl-7b",
            prompt_template="You are a helpful AI assistant. Answer the user's question based on the following context retrieved from a knowledge graph.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        )
        print("Graph RAG pipeline built successfully.")

    def query(self, query_str: str) -> str:
        """Queries the RAG pipeline."""
        if not self.retriever or not self.generator:
            return "Error: The knowledge graph has not been processed yet. Please process a folder first."

        try:
            retrieved_docs = self.retriever.call(input=query_str, k=5)
            context = "\n".join(retrieved_docs)

            answer = self.generator.call(query=query_str, context=context)
            return answer
        except Exception as e:
            return f"An error occurred during query processing: {e}"
