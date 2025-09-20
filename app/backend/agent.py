import os
import networkx as nx
from typing import TypedDict, List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from .document_parser import process_document


# Define the state for our graph
class GraphState(TypedDict):
    file_path: str
    doc_type: str
    doc_content: str
    extracted_json: Dict[str, Any]
    graph: nx.Graph
    error: str


# Initialize the VLM
# We use a structured prompt to get JSON output directly from the model
llm = ChatOllama(model="qwen2.5-vl-7b", format="json")

prompt_template = """
You are an expert AI assistant specializing in document analysis and knowledge graph construction.
Analyze the provided document content and extract key entities and concepts.

**Instructions:**
1.  **Identify Entities:** Extract named entities such as People, Organizations, Locations, Dates, and Products.
2.  **Identify Concepts:** Extract key concepts, themes, or topics discussed. These should be abstract ideas or subjects, not specific names. For example, "strategic partnership," "market analysis," "Q3 financial results."
3.  **Format Output:** Respond ONLY with a single JSON object containing two keys: "entities" and "concepts". Each key should have a list of strings as its value.

**Example Output:**
{{
  "entities": ["Apple Inc.", "Tim Cook", "Paris", "Q3 2024"],
  "concepts": ["corporate expansion", "new product launch", "european market strategy"]
}}

**Document Content to Analyze:**
---
{document_content}
---
"""

# Create the LangChain chain for extraction
parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template(template=prompt_template)
extraction_chain = prompt | llm | parser


def ingest_document(state: GraphState) -> GraphState:
    """Node to load and parse the document."""
    file_path = state["file_path"]
    result = process_document(file_path)

    if result["error"]:
        return {**state, "error": result["error"]}

    return {**state, "doc_type": result["type"], "doc_content": result["content"]}


def extract_entities_and_concepts(state: GraphState) -> GraphState:
    """Node to extract information using the VLM."""
    if state.get("error"):
        return state

    content = state["doc_content"]

    try:
        if state["doc_type"] == "image":
            # The VLM needs the image path for multimodal processing
            # (Note: This assumes the VLM can access the path. In our Docker setup, it can.)
            # A more robust way would be to base64 encode, but this is simpler for the demo.
            extracted_data = extraction_chain.invoke(
                {"document_content": f"Image at path: {content}"}
            )
        else:
            extracted_data = extraction_chain.invoke({"document_content": content})

        return {**state, "extracted_json": extracted_data}
    except Exception as e:
        return {**state, "error": f"VLM extraction failed: {e}"}


def update_knowledge_graph(state: GraphState) -> GraphState:
    """Node to update the NetworkX graph with extracted info."""
    if state.get("error"):
        return state

    graph = state.get("graph", nx.Graph())
    data = state.get("extracted_json", {})
    doc_name = os.path.basename(state["file_path"])

    # Add document node
    graph.add_node(doc_name, type="document", color="#87CEEB")

    # Add entity nodes and connect to document
    for entity in data.get("entities", []):
        graph.add_node(entity, type="entity", color="#F5DEB3")
        graph.add_edge(doc_name, entity)

    # Add concept nodes and connect entities to them
    for concept in data.get("concepts", []):
        graph.add_node(concept, type="concept", color="#98FB98")
        # Connect all entities from this doc to this concept
        for entity in data.get("entities", []):
            graph.add_edge(entity, concept)

    return {**state, "graph": graph}


# Define the workflow
workflow = StateGraph(GraphState)
workflow.add_node("ingest", ingest_document)
workflow.add_node("extract", extract_entities_and_concepts)
workflow.add_node("update_graph", update_knowledge_graph)

workflow.set_entry_point("ingest")
workflow.add_edge("ingest", "extract")
workflow.add_edge("extract", "update_graph")
workflow.add_edge("update_graph", END)

# Compile the graph
graph_builder_app = workflow.compile()
