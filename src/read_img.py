"""
Este módulo implementa la lectura de imágenes médicas en diferentes formatos.
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2
from PIL import Image
import pydicom


class ImageReader(ABC):
    """Clase abstracta para la lectura de imágenes."""
    
    @abstractmethod
    def read(self, path: str) -> tuple:
        """
        Lee una imagen y la retorna en formato array y PIL Image.
        
        Args:
            path (str): Ruta de la imagen a leer.
            
        Returns:
            tuple: (numpy.ndarray, PIL.Image)
        """
        pass


class DicomReader(ImageReader):
    """Implementación para lectura de archivos DICOM."""
    
    def read(self, path: str) -> tuple:
        """
        Lee una imagen DICOM.
        
        Args:
            path (str): Ruta del archivo DICOM.
            
        Returns:
            tuple: (numpy.ndarray, PIL.Image)
        """
        try:
            # Leer archivo DICOM
            dcm = pydicom.dcmread(path)
            img_array = dcm.pixel_array
            
            # Normalizar la imagen
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(float)
                if img_array.max() != img_array.min():
                    img_array = ((img_array - img_array.min()) * 255.0 / 
                               (img_array.max() - img_array.min()))
                img_array = img_array.astype(np.uint8)
            
            # Asegurar que la imagen esté en RGB para la visualización
            if len(img_array.shape) == 2:
                img_RGB = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            else:
                img_RGB = img_array
            
            # Crear una imagen PIL para mostrar
            img2show = Image.fromarray(img_RGB)
            
            return img_RGB, img2show
            
        except Exception as e:
            print(f"Error al leer archivo DICOM: {str(e)}")
            raise


class JpgReader(ImageReader):
    """Implementación para lectura de archivos JPG/JPEG."""
    
    def read(self, path: str) -> tuple:
        """
        Lee una imagen JPG/JPEG.
        
        Args:
            path (str): Ruta del archivo JPG.
            
        Returns:
            tuple: (numpy.ndarray, PIL.Image)
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
            
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2show = Image.fromarray(img_RGB)
        return img_RGB, img2show


class ImageReaderFactory:
    """Factory para crear el lector de imágenes apropiado."""
    
    @staticmethod
    def get_reader(file_extension: str) -> ImageReader:
        """
        Retorna el lector apropiado según la extensión del archivo.
        
        Args:
            file_extension (str): Extensión del archivo.
            
        Returns:
            ImageReader: Instancia del lector apropiado.
            
        Raises:
            ValueError: Si la extensión no está soportada.
        """
        readers = {
            'dcm': DicomReader(),
            'jpg': JpgReader(),
            'jpeg': JpgReader(),
            'png': JpgReader()
        }
        
        reader = readers.get(file_extension.lower())
        if reader is None:
            raise ValueError(f"Formato de archivo no soportado: {file_extension}")
        return reader