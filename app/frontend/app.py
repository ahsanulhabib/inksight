import streamlit as st
import requests
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="InkSight Document Intelligence")
st.title("✒️ InkSight: VLM-Powered Document Intelligence")


# --- Helper Functions ---
def visualize_graph(graph_data: dict):
    """Generates and displays a Pyvis graph from NetworkX data."""
    if not graph_data or not graph_data.get("nodes"):
        st.warning("The knowledge graph is empty.")
        return

    G = nx.node_link_graph(graph_data)
    net = Network(
        height="600px",
        width="100%",
        directed=False,
        notebook=True,
        cdn_resources="in_line",
    )

    for node, attrs in G.nodes(data=True):
        net.add_node(
            node,
            label=str(node),
            title=attrs.get("type", "N/A"),
            color=attrs.get("color", "#87CEEB"),
        )

    for u, v in G.edges():
        net.add_edge(u, v)

    net.show_buttons(filter_=["physics"])
    html = net.generate_html(name="kg_viz.html", local=False)
    components.html(html, height=620)


# --- Initialize Session State ---
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "processing_log" not in st.session_state:
    st.session_state.processing_log = "Ready to process."
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# --- Backend URL ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    st.info(
        "The system processes documents from the `data` folder mounted inside the application."
    )

    if st.button("Process Documents", type="primary"):
        with st.spinner("Processing documents... This may take several minutes."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/process-folder", json={"folder_path": "data"}
                )
                response.raise_for_status()
                result = response.json()

                if "error" in result:
                    st.session_state.processing_log = f"Error: {result['error']}"
                    st.error(st.session_state.processing_log)
                else:
                    st.session_state.graph_data = result.get("graph_data")
                    st.session_state.processing_log = result.get(
                        "message", "Processing complete."
                    )
                    st.success(st.session_state.processing_log)
                    st.rerun()

            except requests.exceptions.RequestException as e:
                st.session_state.processing_log = f"Failed to connect to backend: {e}"
                st.error(st.session_state.processing_log)

    st.subheader("Processing Log")
    st.text_area("Log", st.session_state.processing_log, height=100, disabled=True)

# --- Main Content Area ---
if st.session_state.graph_data is None:
    st.info(
        "Welcome to InkSight! Click 'Process Documents' in the sidebar to build your knowledge graph."
    )
else:
    # --- Tabbed Layout for KG and Querying ---
    tab1, tab2 = st.tabs(["🧠 Knowledge Graph", "❓ Query the Graph"])

    with tab1:
        st.subheader("Interactive Knowledge Graph")
        visualize_graph(st.session_state.graph_data)

    with tab2:
        st.subheader("Query the Knowledge Graph")
        query = st.text_input("Ask a question about your documents:", key="query_input")

        if st.button("Ask"):
            if query:
                with st.spinner("Thinking..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/query", json={"query": query}
                        )
                        response.raise_for_status()
                        answer = response.json().get("answer", "No answer received.")
                        st.session_state.query_history.insert(
                            0, {"question": query, "answer": answer}
                        )
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to get answer from backend: {e}")
            else:
                st.warning("Please enter a question.")

        st.markdown("---")
        st.subheader("History")
        for item in st.session_state.query_history:
            with st.expander(f"**Q:** {item['question']}"):
                st.markdown(f"**A:** {item['answer']}")
