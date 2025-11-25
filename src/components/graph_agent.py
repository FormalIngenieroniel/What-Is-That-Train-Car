import networkx as nx
import pickle
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
# Reutilizamos tu generador existente, pero lo llamaremos manualmente
from src.components.generator import client as gemini_client 
from google.genai import types

# Cargar el grafo creado en la ingestión
try:
    with open(config.CHROMA_PERSIST_DIR / "knowledge_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
except:
    G = nx.DiGraph() # Fallback vacío

# --- 1. DEFINIR EL ESTADO ---
class AgentState(TypedDict):
    question: str
    context: List[dict]
    answer: str

# --- 2. NODOS DEL GRAFO (Tools) ---

def search_graph_node(state: AgentState):
    """Busca en el grafo NetworkX navegando por nodos vecinos"""
    query = state["question"].lower()
    print(f"🕸️ Agente explorando grafo para: {query}")
    
    found_files = set()
    
    # Lógica de búsqueda en Grafo:
    # 1. Identificar keywords en la query que coincidan con nodos atributos
    # 2. Viajar de esos atributos a los archivos conectados
    
    keywords = [node for node in G.nodes if node in query and G.nodes[node].get('type') != 'file']
    
    for key in keywords:
        # Obtener vecinos (archivos conectados a este keyword)
        neighbors = G.neighbors(key)
        for n in neighbors:
            if G.nodes[n].get('type') == 'file':
                found_files.add(n)
    
    # Formatear contexto
    context_list = []
    for filename in found_files:
        node_data = G.nodes[filename]
        context_list.append({
            "filename": filename,
            "description": node_data['description'],
            "image_path": node_data['path'],
            "relevance_score": 1.0 # En grafos binarios, si está conectado es relevante
        })
    
    if not context_list:
        # Fallback: si no encuentra por grafo, devuelve mensaje vacío
        print("⚠️ No se encontraron conexiones en el grafo.")
        
    return {"context": context_list}

def generate_answer_node(state: AgentState):
    """Genera la respuesta usando Gemini con el contexto del grafo"""
    context = state["context"]
    query = state["question"]
    
    if not context:
        return {"answer": "No encontré información relacionada en el grafo de conocimiento."}

    # Preparamos el prompt igual que en tu generador original
    context_text = "\n".join([f"- Archivo {c['filename']}: {c['description']}" for c in context])
    
    # Usamos la imagen del primer resultado
    from PIL import Image
    import PIL
    
    try:
        img_path = context[0]['image_path']
        image = Image.open(img_path)
        
        prompt = f"""
        Pregunta: {query}
        Contexto del Grafo: {context_text}
        
        Responde basándote en la imagen y el texto. Indica qué nodo/archivo usaste.
        """
        
        response = gemini_client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[image, prompt]
        )
        return {"answer": response.text}
        
    except Exception as e:
        return {"answer": f"Error generando respuesta: {e}"}

# --- 3. CONSTRUCCIÓN DE LANGGRAPH ---
workflow = StateGraph(AgentState)

# Agregar nodos
workflow.add_node("search_graph", search_graph_node)
workflow.add_node("generate", generate_answer_node)

# Definir flujo
workflow.set_entry_point("search_graph")
workflow.add_edge("search_graph", "generate")
workflow.add_edge("generate", END)

# Compilar aplicación
graph_app = workflow.compile()