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

    # Usaremos solo el contexto más relevante para el prompt multimodal
    best_context = retrieved_context[0]
    
    try:
        # 1. Preparar la imagen y el texto
        image_path = best_context['image_path']
        image = Image.open(image_path)
        
        context_description = best_context['description']
        filename = best_context['filename']
        
        # 2. Formular el prompt final que incluye el contexto de la búsqueda
        formatted_prompt = f"""
        PREGUNTA DEL USUARIO: "{user_prompt}"

        CONTEXTO RECUPERADO (Texto):
        Descripción del archivo '{filename}' (Score de Relevancia: {best_context['relevance_score']:.2f}):
        {context_description}

        IMAGEN RECUPERADA:
        (La imagen adjunta es el archivo '{filename}')

        Responde la pregunta basándote en esta información y en la IMAGEN.
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