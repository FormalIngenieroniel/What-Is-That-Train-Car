# components/generator.py
import os
import sys
from google import genai
from google.genai import types
from PIL import Image

# Añadir el directorio raíz al path para importar config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

# Inicializar el cliente de Gemini
client = genai.Client(api_key=config.GEMINI_API_KEY)

# Prompt de sistema para dirigir el comportamiento de Gemini
SYSTEM_PROMPT = """
Eres un experto en catalogación de vagones de tren y en el sistema de búsqueda RAG.
Tu tarea es analizar la PREGUNTA del usuario y el CONTEXTO que se te proporciona (incluyendo imágenes y texto descriptivo) 
para ofrecer una respuesta precisa, detallada y bien fundamentada.

REGLAS:
1. Siempre basa tu respuesta únicamente en la información contenida en el CONTEXTO.
2. Menciona explícitamente el nombre del archivo de la imagen que usaste para responder (ej: '01.jpg').
3. Si el contexto es insuficiente o no es relevante, indícalo.
4. Tu objetivo es describir la imagen más relevante y conectar la descripción con la pregunta del usuario.
"""


def generate_response(user_prompt: str, retrieved_context: list):
    """
    Envía el prompt multimodal a Gemini 2.5 Flash.
    
    Args:
        user_prompt (str): Pregunta original del usuario.
        retrieved_context (list): Lista de diccionarios con el contexto recuperado.
        
    Returns:
        str: Respuesta final generada por Gemini.
    """
    if not retrieved_context:
        return "Lo siento, la búsqueda vectorial no encontró información relevante para tu pregunta."

    # 1. Agrupar TODAS las descripciones recuperadas en un solo bloque de texto.
    context_descriptions_text = ""
    for i, context in enumerate(retrieved_context):
        context_descriptions_text += f"--- CONTEXTO {i+1} (Archivo: {context['filename']}, Score: {context['relevance_score']:.2f}) ---\n"
        context_descriptions_text += f"{context['description']}\n\n"
        
    # Usamos la imagen del resultado más relevante, que sigue siendo retrieved_context[0]
    best_context_for_image = retrieved_context[0]
    
    try:
        image_path = best_context_for_image['image_path']
        image = Image.open(image_path)
        filename = best_context_for_image['filename']
        
        # 2. Formular el prompt final
        formatted_prompt = f"""
        PREGUNTA DEL USUARIO: "{user_prompt}"

        CONTEXTO RECUPERADO (Texto - Contiene {len(retrieved_context)} resultados):
        {context_descriptions_text}

        IMAGEN ADJUNTA:
        (La imagen adjunta es el archivo '{filename}', el mejor resultado vectorial. Analiza el texto recuperado para determinar si este archivo O CUALQUIER OTRO CONTEXTO TEXTUAL de la lista responde a la pregunta.)

        Responde la pregunta basándote en la IMAGEN Y/O el CONTEXTO RECUPERADO. Si un contexto de texto es mejor, úsalo, pero siempre menciona el nombre del archivo asociado.
        """
        
        # 3. Construir la solicitud multimodal (Texto + Imagen)
        contents = [
            image,          # La imagen es el primer elemento multimodal
            formatted_prompt # El texto del prompt final
        ]
        
        # 4. Llamada a la API
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error en la generación de Gemini: {e}")
        return "Ocurrió un error al contactar al modelo generador. Revisa tu clave API y la ruta de la imagen."