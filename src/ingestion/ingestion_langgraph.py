import networkx as nx
import pickle
import os
import sys
from pathlib import Path

# Configuración de rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

GRAPH_PATH = config.CHROMA_PERSIST_DIR / "knowledge_graph.gpickle"

def build_graph():
    print("--- 🕸️ Construyendo Grafo de Conocimiento (NetworkX) ---")
    
    # Creamos un grafo dirigido
    G = nx.DiGraph()
    
    image_paths = [config.IMAGE_DIR / f for f in config.IMAGE_FILENAMES]
    descriptions = config.DESCRIPTIONS
    
    for path, desc in zip(image_paths, descriptions):
        filename = path.name
        desc_lower = desc.lower()
        
        # 1. Crear Nodo Central (El Archivo)
        G.add_node(filename, type="file", path=str(path), description=desc)
        
        # 2. Extraer Entidades Simples (Reglas básicas para el ejemplo)
        # En un caso real, usarías un LLM para extraer entidades.
        
        # -- Extracción de Colores --
        colores = ["rojo", "azul", "verde", "amarillo", "gris", "blanco", "negro", "oxidado"]
        for color in colores:
            if color in desc_lower:
                G.add_node(color, type="atributo_color")
                G.add_edge(filename, color, relation="tiene_color")
                G.add_edge(color, filename, relation="es_color_de") # Relación inversa para búsqueda
        
        # -- Extracción de Carga/Tipo --
        cargas = ["petróleo", "neft", "carbón", "madera", "grano", "sellado", "abierto", "cisterna"]
        for carga in cargas:
            if carga in desc_lower:
                G.add_node(carga, type="atributo_carga")
                G.add_edge(filename, carga, relation="transporta_o_es")
                G.add_edge(carga, filename, relation="transportado_por")

    # 3. Guardar el Grafo
    print(f"📊 Nodos creados: {len(G.nodes)}")
    print(f"🔗 Relaciones creadas: {len(G.edges)}")
    
    # Creamos carpeta si no existe (usamos la misma de chroma por comodidad)
    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
    
    with open(GRAPH_PATH, 'wb') as f:
        pickle.dump(G, f)
        
    print(f"✅ Grafo guardado en: {GRAPH_PATH}")

def load_graph():
    """Función helper para cargar el grafo en memoria"""
    with open(GRAPH_PATH, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    build_graph()