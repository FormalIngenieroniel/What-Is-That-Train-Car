# config.py
import os
from pathlib import Path

# --- Variables de Entorno y API Keys ---
# Asegúrate de configurar la clave de Gemini en tu entorno:
# export GEMINI_API_KEY="TU_CLAVE_AQUI"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Rutas de Archivos ---
# Directorio donde están tus imágenes (ajustar si es necesario)
BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "src" / "images"

# --- Configuración de Modelos ---
# Modelo a usar para la Generación (Google)
GEMINI_MODEL = "gemini-2.5-flash"

# Modelo de Embeddings Multimodal (Basado en OpenCLIP/HuggingFace)
# Este modelo genera el vector para la imagen Y el vector para el texto.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# --- Configuración de ChromaDB ---
# Nombre de la Colección (el índice donde se guardan los datos)
CHROMA_COLLECTION_NAME = "vagones_multimodal_clip"
# Cliente: Usaremos el modo persistente (local) para simplificar
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# --- Simulación de Dataset (Reemplazar con tus 13 datos reales) ---
# Se utiliza para la ingesta. Debes asegurar 1 a 1 correspondencia.
IMAGE_FILENAMES = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])

DESCRIPTIONS = [
    "01.jpg: Un vagón de tren de caja cerrada o vagón cubierto de un color rojo oscuro o borgoña, con paneles laterales acanalados y números grandes en blanco, incluyendo el logotipo circular de 'ГРУЗОВАЯ КОМПАНИЯ'. El vagón está sobre rieles bajo un cielo parcialmente nublado, con colinas cubiertas de densa vegetación verde en el fondo, sugiriendo un entorno montañoso. Las ruedas y el chasis son negros.",
    
    "02.jpg: Un vagón de tren de tipo góndola o tolva abierta de un color verde oliva o verde claro industrial, completamente lleno con una gran pila de material granular gris, probablemente grava, carbón o mineral. La parte superior del vagón está abierta. Los números de serie '55638167' están inscritos en el lateral en color verde más claro. El vagón se encuentra en una estación de tren con concreto en primer plano y postes de luz de fondo bajo un cielo blanco y cubierto.",
    
    "03.jpg: Un vagón cisterna de tren de color verde brillante con una línea horizontal blanca que rodea el tanque cilíndrico. Presenta el logotipo de 'ГРУППА КОМПАНИЙ СОДРУЖЕСТВО' y la inscripción 'ЗАО ЕВРОПА-КАЛИНИНГРАД' en mayúsculas blancas. Tiene un pasamanos y escotillas a lo largo de la parte superior y está diseñado para transportar grano (ЗЕРНО) a granel. El tren está bajo un cielo azul brillante.",
    
    "04.jpg: Una vista panorámica de varios vagones de tren de caja abierta o góndola de color gris oscuro o carbón, circulando por una vía férrea. Se trata de un tren de carga largo. Los vagones muestran el gran logo rojo y blanco de 'ФГК' (FGK). La perspectiva es lateral y dinámica, con el cielo azul y cables de electrificación arriba.",
    
    "05.jpg: Dos vagones de tren de caja cerrada o vagón cubierto de un color verde vibrante o brillante, unidos. Los paneles laterales son acanalados y tienen grandes inscripciones blancas, incluido el número '636 73149' y el logo 'РЖД/RZD'. En el lateral se observan varias inscripciones en cirílico y datos técnicos en blanco, y una rueda de freno manual roja en el bogie. El fondo es un cielo azul intenso.",
    
    "06.jpg: Un vagón de tren de góndola o caja abierta de un color rojo oscuro o terracota, con paneles laterales altos y acanalados. Los números grandes '6051 7067' y la inscripción 'КЗХ' (K3X) están en la parte superior. El vagón está sobre una vía con el suelo cubierto de nieve, bajo un cielo azul claro, lo que sugiere un ambiente invernal.",
    
    "07.jpg: Un primer plano de un vagón de tren de góndola abierta de color gris oscuro, con paneles acanalados. En el centro, destaca un gran logotipo rojo y estilizado de 'ФГК'. Las inscripciones y números de serie, incluyendo '620 71246', son blancos. El tren está sobre vías con hierba seca y el cielo gris de fondo.",
    
    "08.jpg: Un vagón de tren de caja cerrada o vagón cubierto de un color azul marino profundo. Presenta grandes números '52605 870' en la parte superior del vagón y varias inscripciones en blanco sobre los paneles. Está sobre rieles con nieve y escarcha alrededor, y se observa una rueda de freno de color rojo en el chasis, similar a un entorno invernal.",
    
    "09.jpg: Un vagón de tren de tipo tolva o vagón de descarga inferior de color gris con una sección grande de color naranja brillante en el centro, que contiene un logotipo en forma de flecha o chevrón apuntando hacia la derecha. Este vagón está diseñado para el transporte de grano (ЗЕРНО) y tiene la inscripción 'РУСГРОТРАНС'. La parte inferior muestra las tolvas de descarga.",
    
    "10.jpg: Un vagón de tren de caja cerrada de color gris oscuro o gris carbón. Presenta inscripciones técnicas detalladas en blanco, incluyendo el número '616 77258' y referencias a estaciones como 'СТ. НОВОМОСКОВСКАЯ-2-МОСК'. La imagen muestra los paneles laterales con estructuras verticales de refuerzo, y una franja roja y parte de otro vagón rojo son visibles a la izquierda.",
    
    "11.jpg: Un vagón de tren de tipo plataforma o vagón de carga abierta de color marrón rojizo u óxido. Es largo y plano, diseñado para transportar maquinaria o artículos grandes. Los paneles laterales son bajos y pueden ser abatibles. Los números grandes '42092288' son visibles. El vagón está en una vía férrea rodeada de vegetación verde y un fondo de bosque denso bajo un cielo nublado.",
    
    "12.jpg: Un vagón cisterna de tren de color rojo oscuro o marrón en el centro, y negro en los extremos del tanque cilíndrico. Lleva el logotipo circular grande de 'ГРУЗОВАЯ КОМПАНИЯ' en blanco y la palabra 'НЕФТЬ' (Petróleo) en el extremo derecho. Está diseñado para transportar líquidos y se encuentra estacionado junto a una estructura blanca de ladrillo bajo un cielo dramático con tonos dorados y nubes.",
    
    "13.jpg: Un vagón de tren de caja cerrada de color naranja brillante, mostrado en un entorno de estudio o renderizado digital. La caja tiene paneles verticales acanalados y varias inscripciones técnicas en blanco, incluyendo el número principal '68965877'. Presenta una rueda de freno manual roja en la parte inferior y está posicionado sobre una vía de ferrocarril con grava limpia y un fondo de estudio gris liso."
]

# Validación básica
if len(IMAGE_FILENAMES) != len(DESCRIPTIONS):
    raise ValueError("El número de imágenes y descripciones en config.py no coincide.")