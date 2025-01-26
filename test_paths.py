# test_paths.py
import os
from pathlib import Path

print("Directorio actual:", os.getcwd())
print("Contenido del directorio actual:", os.listdir())
model_path = Path('models/conv_MLP_84.h5')
print("¿Existe el modelo?:", model_path.exists())
if model_path.exists():
    print("Ruta completa del modelo:", model_path.absolute())