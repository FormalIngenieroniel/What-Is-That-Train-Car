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
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# --- Configuración de ChromaDB ---
# Nombre de la Colección (el índice donde se guardan los datos)
CHROMA_COLLECTION_NAME = "vagones_multimodal_clip"
# Cliente: Usaremos el modo persistente (local) para simplificar
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# --- Simulación de Dataset (Reemplazar con tus 13 datos reales) ---
# Se utiliza para la ingesta. Debes asegurar 1 a 1 correspondencia.
IMAGE_FILENAMES = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])

DESCRIPTIONS = [
    # 01.jpg (Caja Cerrada Roja/Borgoña)
    "Vagón de tren de caja cerrada o vagón cubierto, color rojo oscuro o borgoña. Diseño para transporte de carga seca, mercancías generales o productos a granel protegidos. Logo circular 'ГРУЗОВАЯ КОМПАНИЯ'. Entorno de paisaje montañoso con vegetación verde. Chasis negro.",
    
    # 02.jpg (Góndola Abierta Verde/Grava)
    "Vagón de tipo góndola o caja abierta, color verde oliva industrial, completamente lleno de material granular gris (grava, carbón o mineral). Estructura abierta que facilita la carga y descarga de materiales a granel no perecederos. Estación de tren bajo cielo cubierto.",
    
    # 03.jpg (Tolva/Cisterna Verde/Grano)
    "Vagón cisterna o tolva cubierto de color verde brillante con línea horizontal blanca. Logo de 'СОДРУЖЕСТВО'. Diseñado específicamente para el transporte de 'ЗЕРНО' (grano) a granel, protegiéndolo de los elementos. Escotillas de carga superior. Bajo cielo azul brillante.",
    
    # 04.jpg (Góndola/Caja Abierta Gris/FGK)
    "Varios vagones de tren de caja abierta o góndola, color gris oscuro o carbón. Muestran el gran logo rojo y blanco de 'ФГК' (FGK - Federal Freight). Lados altos e ideales para transporte masivo de materiales a granel como metal, madera o carbón. Tren largo en vía férrea.",
    
    # 05.jpg (Caja Cerrada Verde/RZD) <--- ¡Prefijo eliminado!
    "Dos vagones de tren de caja cerrada o vagón cubierto, color verde vibrante o brillante. Logo 'РЖД/RZD' (Ferrocarriles Rusos). Diseño sellado que protege la carga de la intemperie y robo. Apto para mercancía paletizada o embolsada. Fondo de cielo azul intenso.",
    
    # 06.jpg (Góndola Roja/Nieve)
    "Vagón de tren de góndola o caja abierta, color rojo oscuro o terracota. Se utiliza para el transporte de carga general o a granel. Sobre vías con nieve, sugiriendo un ambiente de operación invernal.",
    
    # 07.jpg (Góndola Gris/FGK)
    "Primer plano de vagón de tren de góndola abierta, color gris oscuro. Gran logotipo rojo y estilizado de 'ФГК' (FGK). Apto para chatarra, mineral o carbón. Sobre vías con hierba seca.",
    
    # 08.jpg (Caja Cerrada Azul Marino)
    "Vagón de tren de caja cerrada o vagón cubierto, color azul marino profundo. Vagón sellado diseñado para mercancías que requieren protección total contra el clima. Sobre rieles con nieve y escarcha. Rueda de freno roja.",
    
    # 09.jpg (Tolva Naranja/Gris/Grano)
    "Vagón de tipo tolva o vagón de descarga inferior, color gris con sección naranja brillante. Diseño tolva que permite la descarga eficiente por gravedad. Ideal para 'ЗЕРНО' (grano) o fertilizantes. Inscripción 'РУСГРОТРАНС'.",
    
    # 10.jpg (Caja Cerrada Gris Oscuro/Novomoskovskaya)
    "Vagón de tren de caja cerrada, color gris oscuro o gris carbón. Referencia a estación 'СТ. НОВОМОСКОВСКАЯ-2-МОСК'. Estructura cerrada que asegura el contenido contra daños.",
    
    # 11.jpg (Plataforma Marrón/Maquinaria)
    "Vagón de tipo plataforma o vagón de carga abierta, color marrón rojizo u óxido. Largo y plano, diseñado para transportar maquinaria pesada, vehículos o contenedores (carga que no necesita protección climática). Paneles laterales bajos. Rodeado de vegetación.",
    
    # 12.jpg (Cisterna Rojo/Petróleo)
    "Vagón cisterna o tanque de tren, color rojo oscuro y negro en los extremos. Lleva la palabra 'НЕФТЬ' (Petróleo). Diseñado específicamente para el transporte de líquidos a granel, como combustible o petróleo crudo. Estacionado junto a estructura blanca de ladrillo.",
    
    # 13.jpg (Caja Cerrada Naranja Brillante/Estudio)
    "Vagón de tren de caja cerrada, color naranja brillante. Modelo de estudio o renderizado digital 3D. Representa un vagón de carga genérico para modelado o simulación."
]

# Validación básica
if len(IMAGE_FILENAMES) != len(DESCRIPTIONS):
    raise ValueError("El número de imágenes y descripciones en config.py no coincide.")