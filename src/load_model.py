"""
Este módulo maneja la carga del modelo de red neuronal.
"""

import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model


class ModelLoader:
    """Clase para cargar y gestionar el modelo de IA."""
    
    _instance = None
    _MODEL_DIR = 'models'  # Directorio de modelos
    _DEFAULT_MODEL = 'conv_MLP_84.h5'  # Nombre del modelo por defecto
    
    def __new__(cls):
        """Implementa el patrón Singleton para el modelo."""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance
    
    def _get_model_path(self, model_name: str = _DEFAULT_MODEL) -> Path:
        """
        Construye la ruta completa al archivo del modelo.
        
        Args:
            model_name (str): Nombre del archivo del modelo
            
        Returns:
            Path: Ruta completa al modelo
        """
        # Encontrar la raíz del proyecto (donde está main.py)
        current_dir = Path(os.getcwd())
        model_path = current_dir / self._MODEL_DIR / model_name
        
        print(f"Buscando modelo en: {model_path}")  # Para depuración
        return model_path
    
    def load_model(self, model_name: str = _DEFAULT_MODEL) -> tf.keras.Model:
        """
        Carga el modelo de red neuronal.
        
        Args:
            model_name (str): Nombre del archivo del modelo
            
        Returns:
            tf.keras.Model: Modelo cargado
            
        Raises:
            FileNotFoundError: Si no se encuentra el archivo del modelo
        """
        if self._model is None:
            model_path = self._get_model_path(model_name)
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"No se encontró el modelo en: {model_path}\n"
                    f"Directorio actual: {os.getcwd()}\n"
                    f"Contenido del directorio models/: {list(Path('models').glob('*'))}"
                )
            
            try:
                print(f"Intentando cargar modelo desde: {model_path}")
                self._model = load_model(str(model_path))
                print("Modelo cargado exitosamente")
            except Exception as e:
                print(f"Error al cargar el modelo: {str(e)}")
                raise
        
        return self._model
    
    def get_model(self) -> tf.keras.Model:
        """
        Retorna el modelo cargado.
        
        Returns:
            tf.keras.Model: Modelo cargado
            
        Raises:
            RuntimeError: Si el modelo no ha sido cargado
        """
        if self._model is None:
            raise RuntimeError("El modelo no ha sido cargado. Llame a load_model primero.")
        return self._model