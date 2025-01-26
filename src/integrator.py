"""
Este módulo integra todos los componentes del sistema.
"""

import os
from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image

from .read_img import ImageReaderFactory
from .preprocess_img import XRayPreprocessor
from .load_model import ModelLoader
from .grad_cam import GradCAM


class PneumoniaDetector:
    """Clase principal que integra todos los componentes del sistema."""
    
    def __init__(self, model_path: str = 'conv_MLP_84.h5'):
        """
        Inicializa el detector de neumonía.
        
        Args:
            model_path (str): Ruta al archivo del modelo.
        """
        self.model_loader = ModelLoader()
        self.model = self.model_loader.load_model(model_path)
        self.preprocessor = XRayPreprocessor()
        self.grad_cam = GradCAM(self.model)
    
    def process_image(self, image_input: Union[str, np.ndarray]) -> Tuple[str, float, np.ndarray]:
        """
        Procesa una imagen y retorna la predicción.
        
        Args:
            image_input: Puede ser una ruta a la imagen (str) o un array numpy con la imagen
            
        Returns:
            tuple: (clase_predicha, probabilidad, imagen_heatmap)
            
        Raises:
            FileNotFoundError: Si no se encuentra la imagen
            ValueError: Si el formato de imagen no está soportado
        """
        if isinstance(image_input, str):
            # Si es una ruta de archivo
            if not Path(image_input).exists():
                raise FileNotFoundError(f"No se encontró la imagen en: {image_input}")
            
            # Obtener la extensión del archivo
            file_extension = Path(image_input).suffix[1:]  # Eliminar el punto
            
            # Crear el lector apropiado y leer la imagen
            reader = ImageReaderFactory.get_reader(file_extension)
            image_array, _ = reader.read(image_input)
        else:
            # Si es un array numpy
            image_array = image_input
        
        # Preprocesar la imagen
        processed_image = self.preprocessor.preprocess(image_array)
        
        # Realizar predicción
        prediction = self.model.predict(processed_image)
        class_idx = np.argmax(prediction[0])
        probability = np.max(prediction[0]) * 100
        
        # Mapear índice a etiqueta
        labels = {0: "bacteriana", 1: "normal", 2: "viral"}
        predicted_class = labels[class_idx]
        
        # Generar heatmap
        heatmap = self.grad_cam.generate_heatmap(processed_image, image_array)
        
        return predicted_class, probability, heatmap