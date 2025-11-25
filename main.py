# main.py
from src.ingestion.ingestion_chroma import load_data_to_chroma
from src.components.retriever import search_chroma
from src.components.generator import generate_response
import config

if __name__ == "__main__":
    print("==============================================")
    print("       🚀 Proyecto Final RAG Multimodal        ")
    print("==============================================")

    # --- 1. Fase de Ingesta (Se ejecuta solo la primera vez) ---
    # Asegúrate de que las imágenes y las descripciones en config.py sean correctas.
    load_data_to_chroma()
    
    print("\n--- 2. Fase de Prueba y RAG ---")
    
    # Ejemplo de Query 1: Búsqueda semántica de imagen vs texto
    query_1 = "Necesito el vagón cisterna que transporta petróleo (NEFT)."
    print(f"\nPregunta: {query_1}")
    
    # A. Recuperación
    context_1 = search_chroma(query_1, n_results=3)
    
    # B. Generación
    respuesta_1 = generate_response(query_1, context_1)
    
    print("\n[Respuesta del Sistema RAG]:")
    print(respuesta_1)
    print("----------------------------------------------")


    # Ejemplo de Query 2: Búsqueda de vagones específicos por color
    query_2 = "Muéstrame el vagón de carga sellado de color azul marino profundo."
    print(f"\nPregunta: {query_2}")
    
    # A. Recuperación
    context_2 = search_chroma(query_2, n_results=3)
    
    # B. Generación
    respuesta_2 = generate_response(query_2, context_2)
    
    print("\n[Respuesta del Sistema RAG]:")
    print(respuesta_2)
    print("==============================================")