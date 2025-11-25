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

# Cargar el modelo CLIP (Large) - Debe ser el mismo de config.py
MODEL_NAME = config.CLIP_MODEL_NAME
print(f"Cargando modelo Multimodal: {MODEL_NAME}...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def get_combined_embedding(image_path: Path, text: str):
    try:
        image = Image.open(image_path)
        
        # 1. Obtener Vector de Imagen (Igual que antes)
        inputs_img = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs_img)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # 2. Obtener Vector de Texto (¡AQUÍ ESTÁ EL CAMBIO!)
        # Agregamos truncation=True y max_length=77
        inputs_txt = processor(
            text=[text], 
            return_tensors="pt", 
            padding=True, 
            truncation=True,    # <--- NUEVO: Cortar si es muy largo
            max_length=77       # <--- NUEVO: Límite de CLIP
        )
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs_txt)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # 3. FUSIÓN (Igual que antes)
        combined_features = (image_features + text_features) / 2.0
        combined_features = combined_features / combined_features.norm(p=2, dim=-1, keepdim=True)

        return combined_features.squeeze().tolist()

    except Exception as e:
        print(f"Error procesando multimodal {image_path}: {e}")
        return None


def load_data_to_chroma():
    print("--- ⚙️ Iniciando Ingesta Multimodal (Fusión Imagen + Texto) ---")
    
    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    
    # Reiniciar colección para asegurar vectores nuevos
    try:
        client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
        print(f"Colección antigua eliminada.")
    except:
        pass 
        
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    image_paths = [config.IMAGE_DIR / f for f in config.IMAGE_FILENAMES]
    descriptions = config.DESCRIPTIONS
    
    embeddings_list = []
    metadatas_list = []
    documents_list = []
    ids_list = []

    print("Generando vectores fusionados (Imagen + Semántica)...")

    for i, (path, desc) in enumerate(zip(image_paths, descriptions)):
        
        # LLAMADA A LA NUEVA FUNCIÓN DE FUSIÓN
        embedding = get_combined_embedding(path, desc)
        
        if embedding:
            embeddings_list.append(embedding)
            metadatas_list.append({
                "filename": path.name, 
                "description": desc, 
                "wagon_type": "cargo_train"
            })
            documents_list.append(desc) 
            ids_list.append(f"wagon_{i+1}")

    if embeddings_list:
        collection.add(
            embeddings=embeddings_list,
            metadatas=metadatas_list,
            documents=documents_list,
            ids=ids_list
        )
        print(f"✅ Ingesta completada. Total de registros: {collection.count()}")
    else:
        print("❌ No se pudieron generar embeddings.")

if __name__ == "__main__":
    load_data_to_chroma()