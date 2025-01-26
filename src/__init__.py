"""
Paquete principal para la aplicación de detección de neumonía.
"""

from .read_img import ImageReaderFactory
from .preprocess_img import XRayPreprocessor
from .load_model import ModelLoader
from .grad_cam import GradCAM
from .integrator import PneumoniaDetector

__version__ = '1.0.0'
__author__ = 'David Plaza'