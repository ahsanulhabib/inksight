import os
import networkx as nx
from fastapi import FastAPI
from pydantic import BaseModel
from .agent import graph_builder_app
from .rag_pipeline import GraphRAG

app = FastAPI()

# In-memory storage for the demo
# A new graph and RAG pipeline are created each time a folder is processed
knowledge_graph: nx.Graph = nx.Graph()
rag_pipeline: GraphRAG = GraphRAG()


class FolderRequest(BaseModel):
    folder_path: str = "data"


class QueryRequest(BaseModel):
    query: str


@app.post("/process-folder")
async def process_folder(request: FolderRequest):
    global knowledge_graph, rag_pipeline

    folder_path = request.folder_path
    if not os.path.isdir(folder_path):
        return {"error": "Invalid folder path provided."}

    # Reset the graph for a new processing run
    current_graph = nx.Graph()
    files_to_process = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    for file_path in files_to_process:
        print(f"Processing: {file_path}")
        inputs = {"file_path": file_path, "graph": current_graph}
        result = graph_builder_app.invoke(inputs)

        if result.get("graph"):
            current_graph = result["graph"]
        else:
            print(
                f"Warning: Could not process {file_path}. Error: {result.get('error')}"
            )

    knowledge_graph = current_graph

    # Build the RAG pipeline with the new graph
    rag_pipeline = GraphRAG(graph=knowledge_graph)

    # Convert graph to a JSON-serializable format for the frontend
    graph_json = nx.node_link_data(knowledge_graph)
    return {
        "message": f"Processed {len(files_to_process)} files.",
        "graph_data": graph_json,
    }


@app.post("/query")
async def query_knowledge_graph(request: QueryRequest):
    global rag_pipeline
    if not rag_pipeline or not rag_pipeline.generator:
        return {
            "answer": "The RAG pipeline is not initialized. Please process a folder first."
        }

    answer = rag_pipeline.query(request.query)
    return {"answer": answer}
