"""
Este módulo implementa el preprocesamiento de imágenes médicas.
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod


class ImagePreprocessor(ABC):
    """Clase abstracta para el preprocesamiento de imágenes."""
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen.
        
        Args:
            image (numpy.ndarray): Imagen a preprocesar.
            
        Returns:
            numpy.ndarray: Imagen preprocesada.
        """
        pass


class XRayPreprocessor(ImagePreprocessor):
    """Implementación del preprocesamiento para radiografías."""
    
    def __init__(self, target_size: tuple = (512, 512)):
        """
        Inicializa el preprocesador.
        
        Args:
            target_size (tuple): Tamaño objetivo de la imagen (altura, ancho).
        """
        self.target_size = target_size
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen de rayos X.
        
        Args:
            image (numpy.ndarray): Imagen a preprocesar.
            
        Returns:
            numpy.ndarray: Imagen preprocesada en formato batch.
        """
        # Redimensionar
        image = cv2.resize(image, self.target_size)
        
        # Convertir a escala de grises
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        image = clahe.apply(image)
        
        # Normalizar
        image = image / 255.0
        
        # Convertir a formato batch
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return image