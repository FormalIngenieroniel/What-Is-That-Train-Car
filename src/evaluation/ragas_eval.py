# evaluation/ragas_eval.py
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_recall
import os
import sys

# Añadir el directorio raíz al path para importar config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from components.retriever import search_chroma
from components.generator import generate_response

# --- Generación de Datos de Prueba (Simulación) ---
# NOTA: Para un proyecto real, generarías estas preguntas/respuestas automáticamente
# con un LLM o las escribirías manualmente basadas en tus 13 imágenes.

def run_evaluation():
    
    # Define tus preguntas de prueba
    test_questions = [
        "¿Qué tipo de vagón es el de color rojo oscuro y negro diseñado para transportar líquidos?",
        "¿Cuál es la función del vagón verde que tiene una banda blanca horizontal?",
        "Describe el vagón de carga de color gris oscuro con el logo rojo de ФГК."
    ]
    
    # Crea las respuestas ideales (ground truth)
    ground_truths = [
        "Es un vagón cisterna usado para transportar NEFT (petróleo), identificado con el logo de ГРУЗОВАЯ КОМПАНИЯ.",
        "Es un vagón cisterna o tolva cubierto para transportar ZERNO (grano), parte del grupo СОДРУЖЕСТВО.",
        "Es un vagón góndola o caja abierta, acanalado, usado para carga general."
    ]

    # --- Pipeline RAG para RAGAS ---
    data = {
        'question': [],
        'answer': [],
        'contexts': [], # Contextos recuperados por tu sistema
        'ground_truths': ground_truths
    }

    for q in test_questions:
        # 1. Recuperación
        retrieved_context = search_chroma(q, n_results=3)
        contexts = [c['description'] for c in retrieved_context]
        
        # 2. Generación
        generated_answer = generate_response(q, retrieved_context)
        
        # Almacenamiento
        data['question'].append(q)
        data['answer'].append(generated_answer)
        data['contexts'].append(contexts)

    # 3. Crear el Dataset de RAGAS
    dataset = Dataset.from_dict(data)

    # 4. Definir las métricas y evaluar
    result = evaluate(
        dataset, 
        metrics=[faithfulness, answer_relevance, context_recall],
        # Especificar el modelo para la evaluación de RAGAS
        llm=f"gemini/{config.GEMINI_MODEL}" 
    )

    # 5. Imprimir resultados
    print("\n================== 📊 Resultados RAGAS ==================")
    print(result)
    print("\nResultados en formato DataFrame:")
    print(result.to_pandas())
    print("=========================================================")

if __name__ == "__main__":
    run_evaluation()