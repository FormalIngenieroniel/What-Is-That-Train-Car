# src/components/retriever.py
import chromadb
from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path
import os
import sys

# Añadir el directorio raíz al path para importar config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

# Inicializar componentes CLIP (mismo modelo Large que en la ingesta)
MODEL_NAME = config.CLIP_MODEL_NAME
print(f"🔄 Cargando modelo de búsqueda: {MODEL_NAME}...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def text_to_clip_embedding(text: str):
    """
    Convierte un texto (query) en un vector CLIP NORMALIZADO.
    Esto es crucial para que coincida con los vectores normalizados de la ingesta.
    """
    try:
        # Procesar solo texto
        inputs = processor(text=[text], images=None, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            # Obtener features
            text_features = model.get_text_features(**inputs)
            
            # --- PASO CLAVE: NORMALIZACIÓN ---
            # Esto alinea la magnitud del vector con los almacenados en ChromaDB
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        return text_features.squeeze().tolist()
    except Exception as e:
        print(f"Error generando embedding para query: {e}")
        return []


def search_chroma(query_text: str, n_results: int = 3):
    """
    Busca los embeddings multimodales más cercanos al vector del query textual.
    """
    print(f"🔍 Buscando '{query_text}' en ChromaDB...")
    
    # 1. Conexión a ChromaDB
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
        collection = client.get_collection(name=config.CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error de conexión: Asegúrate de haber ejecutado la ingesta primero.\nDetalle: {e}")
        return []

    # 2. Generar el embedding NORMALIZADO del texto de búsqueda
    query_vector = text_to_clip_embedding(query_text)
    
    if not query_vector:
        print("❌ No se pudo generar el vector de búsqueda.")
        return []

    # 3. Ejecutar la búsqueda vectorial
    # Usamos query_embeddings para comparar vector vs vector (768 dimensiones)
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )
    except Exception as e:
        print(f"❌ Error durante la consulta a ChromaDB: {e}")
        return []
    
    # 4. Formatear el contexto recuperado
    context_list = []
    
    if results['ids']:
        # Los resultados vienen anidados, iteramos sobre el primer (y único) query
        for metadata, document, distance in zip(results['metadatas'][0], results['documents'][0], results['distances'][0]):
            
            # Convertir distancia (L2 o Cosine) a un score de relevancia aproximado (0 a 1)
            # Nota: ChromaDB por defecto usa L2 (Euclidean Squared). 
            # Distancias más bajas = Mayor similitud.
            relevance = max(0, 1 - distance) 

            context_list.append({
                "filename": metadata['filename'],
                "description": document,
                "relevance_score": relevance,
                "image_path": str(config.IMAGE_DIR / metadata['filename'])
            })
            
        print(f"✅ Recuperados {len(context_list)} resultados.")
    else:
        print("⚠️ No se encontraron resultados.")

    return context_list