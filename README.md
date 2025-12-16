# 🚂 Multimodal RAG Comparison – Vector DB vs. Knowledge Graph (Train Wagons)

This project implements a **Multimodal Retrieval-Augmented Generation (RAG)** system designed to identify, retrieve, and describe specific train wagons based on visual and textual data.

It serves as a technical benchmark to compare two distinct retrieval architectures:
1.  **Vector Search**: Using **ChromaDB** with **CLIP embeddings** for semantic similarity.
2.  **Knowledge Graph**: Using **LangGraph** and **NetworkX** to map relationships between wagon attributes (color, cargo, type).

The system uses **Google Gemini 2.5 Flash** as the generative engine and uses **Ragas** to evaluate the performance of both approaches using metrics like Faithfulness and Answer Similarity.

---

## 🤖 Introduction to Multimodal RAG Architectures

**Retrieval-Augmented Generation (RAG)** allows LLMs to answer questions based on private data. When dealing with **Multimodal Data** (Images + Text), the retrieval strategy becomes critical. This project explores two paradigms:

-   **Vector-Based RAG**: Converts images and text into high-dimensional vectors (embeddings). Retrieval is based on mathematical distance (similarity). It is excellent for vague or semantic queries.
-   **Graph-Based RAG**: Structures data as nodes and edges (e.g., `Wagon` --*transports*--> `Oil`). Retrieval is based on traversing relationships. It is excellent for structured, logic-based queries.

In this project, the **Agent** acts as a railway expert:
-   **Perception** → CLIP analyzes images; Graph Agent extracts keywords.
-   **Decision-making** → LangGraph or Vector Retriever selects the best context.
-   **Action** → Gemini generates a precise description of the specific wagon (e.g., "The red tanker carrying NEFT").

---

## 🚀 Features

-   📷 **Multimodal Ingestion**: Processes images (`.jpg`) and textual descriptions simultaneously.
-   🧠 **CLIP Embeddings**: Uses `openai/clip-vit-large-patch14` to map text and images into a shared vector space.
-   🗄️ **ChromaDB Integration**: Stores normalized vectors for efficient semantic search.
-   🕸️ **Knowledge Graph Construction**: Builds a NetworkX graph linking files to attributes (Colors, Cargo Types like "Grain", "Coal", "Oil").
-   🤖 **LangGraph Agent**: An intelligent agent that navigates the graph nodes to find relevant wagon contexts.
-   📊 **Ragas Evaluation**: Automated benchmarking pipeline to measure **Faithfulness**, **Answer Similarity**, and **Context Precision**.
-   ⚡ **Gemini 2.5 Flash**: State-of-the-art multimodal generation.

---

## 💻 Code Workflow

### **`config.py`**
-   Central configuration file.
-   Defines paths, model names (`gemini-2.5-flash`, `clip-vit`), and API keys.
-   Contains the **Ground Truth Dataset**: A dictionary of 13 wagon images and their detailed descriptions.

### **`src/ingestion/`**
-   **`ingestion_chroma.py`**: Loads images/text, chunks descriptions using `RecursiveCharacterTextSplitter`, creates CLIP embeddings, and persists them in **ChromaDB**.
-   **`ingestion_langgraph.py`**: Parses descriptions to extract entities (Colors: *Red, Green*; Cargo: *Neft, Grain*) and builds a **NetworkX** graph (`knowledge_graph.gpickle`).

### **`src/components/`**
-   **`retriever.py`**: Handles **Vector Search**. Converts the user query into a CLIP vector and finds the nearest neighbors in ChromaDB.
-   **`graph_agent.py`**: Handles **Graph Search**. Uses **LangGraph** to define a workflow that searches graph nodes based on query keywords and retrieves connected file paths.
-   **`generator.py`**: Receives context (text + image path) and prompts Gemini to answer the user's question.

### **`src/evaluation/`**
-   **`ragas_eval.py`**: Runs the evaluation pipeline on the Vector approach.
-   **`evaluation_graph.py`**: Runs the evaluation pipeline on the Graph approach.
-   Both scripts generate CSV reports (`resultados_real_chroma.csv`, `resultados_real_graph.csv`) comparing the output against Ground Truth.

---

## 🧩 How It Works

### **Approach A: Vector Search (ChromaDB)**
1.  **Ingestion**: Images and text are converted to vectors.
2.  **Query**: "Show me the green wagon carrying grain."
3.  **Retrieval**: The system calculates the cosine similarity between the query vector and stored vectors.
4.  **Result**: Returns the wagon with the highest mathematical similarity (e.g., `03.jpg`).

### **Approach B: Graph Search (LangGraph)**
1.  **Construction**: Nodes are created for files (e.g., `12.jpg`) and Attributes (e.g., `Red`, `Oil/Neft`). Edges define relationships (`12.jpg` --*has_color*--> `Red`).
2.  **Query**: "I need the tanker carrying Oil."
3.  **Traversal**: The agent identifies the node `Oil`, finds all connected `File` nodes, and filters by `Tanker`.
4.  **Result**: Returns the specific file connected to those logical attributes.

---

## 🧠 Evaluation Metrics (Ragas)

To scientifically compare both methods, we use the **Ragas** framework with a "Judge LLM" (Gemini Flash Lite):

-   **Faithfulness**: Does the answer rely *only* on the retrieved context?
-   **Answer Similarity**: How close is the generated answer to the human-defined ground truth?
-   **Context Precision**: Did the retrieval system find the correct image/description at the top of the list?

### **Example Scenario**
*Query:* "Muéstrame el vagón de carga sellado de color azul marino profundo."

* **Ground Truth**: Image `08.jpg` (Blue Boxcar).
* **Vector System**: Might retrieve `08.jpg` and `01.jpg` based on color similarity.
* **Graph System**: Looks for node `Blue` and node `Sealed`, traversing directly to `08.jpg`.

---

## 📊 Project Status

-   ✅ **Vector Ingestion & Retrieval**: Fully operational with CLIP + ChromaDB.
-   ✅ **Graph Construction**: NetworkX graph built with entity extraction rules.
-   ✅ **LangGraph Agent**: Functional workflow for graph-based retrieval.
-   ✅ **Evaluation**: Ragas pipeline active for benchmarking.
-   🔄 **Future Improvements**:
    -   Implement LLM-based entity extraction for graph building (instead of rule-based).
    -   Hybrid Search (combining Vector + Graph scores).

---

## 👨‍💻 Author

Project developed by **Daniel Bernal**.
