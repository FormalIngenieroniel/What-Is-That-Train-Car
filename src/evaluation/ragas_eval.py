import os
import sys
import warnings

# --- 1. CONFIGURACIÓN INICIAL Y PARCHES ---
# Engañamos a la validación de Pydantic
os.environ["OPENAI_API_KEY"] = "sk-no-key-needed"
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from datasets import Dataset

# --- 2. IMPORTS DE RAGAS (NOMBRES ACTUALIZADOS) ---
# Intentamos importar con los nombres nuevos de la versión 0.2+
try:
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,  # ANTES: AnswerRelevance
        ContextPrecision,
        ContextRecall,
        AnswerSimilarity
    )
    # Mapeo por si acaso tu código antiguo espera el nombre viejo
    AnswerRelevance = ResponseRelevancy 
    
except ImportError as e:
    print(f"⚠️ Error importando métricas: {e}")
    print("Intentando importar desde rutas internas como respaldo...")
    # Fallback para versiones híbridas/viejas
    from ragas.metrics._faithfulness import Faithfulness
    try:
        from ragas.metrics._answer_relevance import ResponseRelevancy as AnswerRelevance
    except ImportError:
        # Si falla, intentamos el nombre viejo
        from ragas.metrics._answer_relevance import AnswerRelevance
        
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._answer_similarity import AnswerSimilarity

from ragas import evaluate

# Imports de LangChain y Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuración de Rutas ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# --- Imports Locales (Mockeados si fallan para que puedas probar el script) ---
try:
    import config
    from src.components.retriever import search_chroma
    from src.components.generator import generate_response
except ImportError:
    print("⚠️ Usando Mocks para componentes locales faltantes")
    class ConfigMock: GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    config = ConfigMock()
    def search_chroma(q, n_results=3): return [{'description': 'Contexto simulado del vagón'}]
    def generate_response(q, items): return "Respuesta simulada del vagón"

# --- 3. CONFIGURACIÓN DE MODELOS ---

# LLM Juez (Gemini)
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=config.GEMINI_API_KEY,
    temperature=0,
    # Añadimos estos parámetros para mejorar estabilidad:
    transport="rest", 
    client_options={"api_endpoint": "generativelanguage.googleapis.com"},
)

# Embeddings (HuggingFace local)
print("Cargando Embeddings HuggingFace...")
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def run_evaluation():
    print("\n--- 📊 Iniciando Evaluación RAGAS ---")

    # 1. Datos de prueba
    test_data = [
        {
            "question": "Necesito el vagón cisterna que transporta petróleo (NEFT).",
            "ground_truth": "El vagón es el archivo 12.jpg. Es un vagón cisterna rojo oscuro."
        },
        {
            "question": "Muéstrame el vagón de carga sellado de color azul marino profundo.",
            "ground_truth": "El vagón es el archivo 08.jpg. Es un vagón de caja cerrada azul marino."
        }
    ]

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # 2. Generación
    for item in test_data:
        q = item["question"]
        print(f"Generando para: {q}")
        
        retrieved_items = search_chroma(q, n_results=3)
        context_strings = [str(item.get('description', '')) for item in retrieved_items]
        generated_answer = generate_response(q, retrieved_items)

        questions.append(q)
        answers.append(generated_answer)
        contexts.append(context_strings)
        ground_truths.append(item["ground_truth"])

    # 3. Dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # 4. Definición de Métricas (CORREGIDO)
    print("Configurando métricas...")
    
    # En Ragas v0.2+, instanciamos las métricas SIN argumentos
    metrics_to_run = [
        Faithfulness(),
        AnswerRelevance(), # O ResponseRelevancy
        ContextPrecision(),
        ContextRecall(),
        AnswerSimilarity() 
    ]

    # 5. Evaluación
    print("Calculando scores (esto puede tardar)...")
    
    # Pasamos el LLM y los Embeddings AQUÍ. Ragas los inyecta a las métricas.
    results = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        llm=gemini_llm,       # <--- Se pasa globalmente
        embeddings=hf_embeddings # <--- Se pasa globalmente
    )

    # 6. Resultados
    print("\n================== 📈 Resultados ==================")
    df_results = results.to_pandas()
    
    # Ajuste de nombres de columnas (AnswerRelevance a veces devuelve answer_relevancy)
    cols_to_show = ['question', 'faithfulness', 'answer_relevancy', 'context_precision', 'answer_similarity']
    available_cols = [c for c in cols_to_show if c in df_results.columns]
    
    print(df_results[available_cols].round(3))
    
    print("\nPromedios Globales:")
    print(results)
    
    df_results.to_csv("resultados_evaluacion.csv", index=False)
    print("\nGuardado en 'resultados_evaluacion.csv'")

if __name__ == "__main__":
    run_evaluation()