import sys
import os
import pandas as pd
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerSimilarity, ResponseRelevancy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
from src.components.graph_agent import graph_app # Importamos el Agente LangGraph

# --- Configuración Ragas (Igual que antes) ---
os.environ["OPENAI_API_KEY"] = "no-key"
ragas_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=config.GEMINI_API_KEY)
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

test_data = [
    {
        "question": "Necesito el vagón cisterna que transporta petróleo.",
        "ground_truth": "Es el archivo 12.jpg, vagón cisterna rojo oscuro para petróleo."
    },
    {
        "question": "Vagón azul marino sellado.",
        "ground_truth": "Archivo 08.jpg, vagón de carga cerrado azul."
    }
]

def run_graph_evaluation():
    print("--- 🕸️ Iniciando Evaluación con LangGraph + NetworkX ---")
    
    questions, answers, contexts, gts = [], [], [], []
    
    for item in test_data:
        q = item["question"]
        print(f"Procesando: {q}")
        
        # INVOCACIÓN A LANGGRAPH
        # Pasamos el estado inicial
        result_state = graph_app.invoke({"question": q, "context": [], "answer": ""})
        
        generated_ans = result_state["answer"]
        retrieved_ctx = [c['description'] for c in result_state['context']]
        
        questions.append(q)
        answers.append(generated_ans)
        contexts.append(retrieved_ctx)
        gts.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": gts
    })
    
    print("🚀 Evaluando con Ragas...")
    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerSimilarity(), ResponseRelevancy()],
        llm=ragas_llm,
        embeddings=hf_embeddings
    )
    
    print(results)
    results.to_pandas().to_csv("resultados_graph.csv", index=False)

if __name__ == "__main__":
    run_graph_evaluation()