"""
Tests para el módulo de preprocesamiento de imágenes.
"""

import pytest
import numpy as np
import cv2
from src.preprocess_img import XRayPreprocessor


@pytest.fixture
def sample_image():
    """Fixture que genera una imagen de prueba."""
    # Crear una imagen sintética de 100x100 con valores aleatorios
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


def test_preprocessor_output_shape(sample_image):
    """Prueba que la forma de la imagen de salida sea correcta."""
    preprocessor = XRayPreprocessor(target_size=(512, 512))
    processed_image = preprocessor.preprocess(sample_image)
    
    # Verificar las dimensiones del batch
    assert processed_image.shape == (1, 512, 512, 1)


def test_preprocessor_normalization(sample_image):
    """Prueba que la normalización se realice correctamente."""
    preprocessor = XRayPreprocessor()
    processed_image = preprocessor.preprocess(sample_image)
    
    # Verificar que los valores estén entre 0 y 1
    assert np.min(processed_image) >= 0.0
    assert np.max(processed_image) <= 1.0