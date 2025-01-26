"""
Configuración para las pruebas pytest.
"""

import os
import sys
from pathlib import Path

# Obtener el directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent

# Añadir el directorio raíz al path de Python
sys.path.insert(0, str(ROOT_DIR))