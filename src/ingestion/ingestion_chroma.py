# ingestion/ingestion_chroma.py
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path
import os
import sys

# Añadir el directorio raíz al path para importar config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

# Cargar el modelo CLIP (OpenCLIP architecture)
MODEL_NAME = config.CLIP_MODEL_NAME
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def multimodal_clip_embedding_function(image_path: Path):
    """
    Genera el vector de embedding de una imagen usando el modelo CLIP.
    """
    try:
        image = Image.open(image_path)
        # El procesador solo necesita la imagen para generar su embedding
        inputs = processor(images=image, return_tensors="pt")
        
        # Generar el embedding de la imagen
        with torch.no_grad():
            # Usar get_image_features para obtener el vector de la imagen
            image_features = model.get_image_features(**inputs)
        
        # Convertir a lista de Python para ChromaDB
        return image_features.squeeze().tolist()
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None


def load_data_to_chroma():
    """
    Ejecuta el pipeline completo de ingesta: vectorización y almacenamiento en ChromaDB.
    """
    print("--- ⚙️ Iniciando Ingesta en ChromaDB (Patrón 2: CLIP) ---")
    
    # 1. Inicializar Cliente ChromaDB (Persistente)
    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    
    # Eliminar la colección anterior si existe (para empezar de cero)
    try:
        client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
        print(f"Colección antigua '{config.CHROMA_COLLECTION_NAME}' eliminada.")
    except:
        pass # No hacer nada si no existe
        
    # 2. Crear una nueva colección
    # No definimos una función de embedding aquí, ya que insertaremos los vectores precalculados
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    # 3. Preparación de datos
    image_paths = [config.IMAGE_DIR / f for f in config.IMAGE_FILENAMES]
    descriptions = config.DESCRIPTIONS
    
    embeddings_list = []
    metadatas_list = []
    documents_list = []
    ids_list = []

    # 4. Bucle de vectorización y recolección
    for i, (path, desc) in enumerate(zip(image_paths, descriptions)):
        
        # Calcular embedding de la imagen (la clave del Patrón 2)
        embedding = multimodal_clip_embedding_function(path)
        
        if embedding:
            embeddings_list.append(embedding)
            # Metadatos esenciales para la recuperación y el RAG
            metadatas_list.append({"filename": path.name, "description": desc, "wagon_type": "cargo_train"})
            # Guardamos la descripción como documento para que Gemini pueda leerla
            documents_list.append(desc) 
            ids_list.append(f"wagon_{i+1}")

    # 5. Añadir a ChromaDB
    if embeddings_list:
        collection.add(
            embeddings=embeddings_list,
            metadatas=metadatas_list,
            documents=documents_list,
            ids=ids_list
        )
        print(f"✅ Ingesta completada. Total de registros: {collection.count()}")
    else:
        print("❌ No se pudieron generar embeddings. Revisar archivos e instalación de PyTorch.")


if __name__ == "__main__":
    # Asegúrate de que el directorio de imágenes exista
    if not os.path.isdir(config.IMAGE_DIR):
        print(f"Error: La carpeta de imágenes no existe en {config.IMAGE_DIR}")
        exit()
        
    load_data_to_chroma()