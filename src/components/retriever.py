# components/retriever.py
import chromadb
from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path
import os
import sys


# Añadir el directorio raíz al path para importar config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

# Inicializar componentes CLIP (debe ser el mismo modelo que en la ingesta)
MODEL_NAME = config.CLIP_MODEL_NAME
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def text_to_clip_embedding(text: str):
    """
    Convierte un texto (query) en un vector CLIP para buscar contra embeddings de imagen.
    """
    # Procesar solo texto
    inputs = processor(text=[text], images=None, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # Usar get_text_features para obtener el vector del texto
        text_features = model.get_text_features(**inputs)
    
    return text_features.squeeze().tolist()


def search_chroma(query_text: str, n_results: int = 3):
    """
    Busca los embeddings de imagen más cercanos al vector del query textual.
    """
    print(f"🔍 Buscando '{query_text}' en ChromaDB...")
    
    # 1. CONEXIÓN (¡Ahora la establecemos dentro de la función!)
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
        # Usamos get_collection. Si la ingesta no corrió, esto fallará, 
        # pero es lo que queremos para la prueba de concepto después de la ingesta.
        collection = client.get_collection(name=config.CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error de conexión/colección: La ingesta debe ejecutarse primero. Detalle: {e}")
        return [] # Retorna lista vacía si falla la conexión
    
    # ... (el resto del código de búsqueda queda igual)
    # 2. Generar el embedding del texto
    query_vector = text_to_clip_embedding(query_text)
    
    # 3. Ejecutar la búsqueda vectorial
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=['metadatas', 'documents', 'distances']
    )
    
    # 3. Formatear el contexto recuperado
    context_list = []
    
    # Los resultados vienen anidados, iteramos sobre el primer (y único) query
    for metadata, document, distance in zip(results['metadatas'][0], results['documents'][0], results['distances'][0]):
        context_list.append({
            "filename": metadata['filename'],
            "description": document,  # Es la descripción del vagón
            "relevance_score": 1 - distance, # Distancia (menor es mejor)
            "image_path": str(config.IMAGE_DIR / metadata['filename']) # Ruta local para Gemini
        })
        
    print(f"✅ Recuperados {len(context_list)} resultados.")
    return context_list