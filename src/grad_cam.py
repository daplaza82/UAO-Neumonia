"""
Este módulo implementa el algoritmo Grad-CAM para visualización.
"""

import numpy as np
import cv2
import tensorflow as tf

# Desactivar ejecución eager
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import backend as K


class GradCAM:
    """Implementación del algoritmo Grad-CAM."""
    
    def __init__(self, model: tf.keras.Model, layer_name: str = "conv10_thisone"):
        """
        Inicializa el visualizador Grad-CAM.
        
        Args:
            model (tf.keras.Model): Modelo de red neuronal.
            layer_name (str): Nombre de la capa convolucional de interés.
        """
        self.model = model
        self.layer_name = layer_name
        
        # Verificar si la capa existe
        try:
            self.layer = self.model.get_layer(layer_name)
        except ValueError as e:
            print(f"Error: {str(e)}")
            print("Capas disponibles:")
            for layer in self.model.layers:
                print(f"- {layer.name}")
            raise
    
    def generate_heatmap(self, processed_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Genera un mapa de calor usando Grad-CAM.
        
        Args:
            processed_image (numpy.ndarray): Imagen preprocesada en formato batch.
            original_image (numpy.ndarray): Imagen original para superposición.
            
        Returns:
            numpy.ndarray: Imagen con el mapa de calor superpuesto.
        """
        # Obtener predicción
        preds = self.model.predict(processed_image)
        class_idx = np.argmax(preds[0])
        
        # Obtener la salida de la última capa convolucional
        last_conv_layer = self.model.get_layer(self.layer_name)
        
        # Obtener los gradientes de la clase con respecto a la última capa convolucional
        class_output = self.model.output[:, class_idx]
        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        
        # Función para obtener valores
        iterate = K.function([self.model.input],
                           [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate(processed_image)
        
        # Aplicar pesos a los mapas de características
        for i in range(pooled_grads_value.shape[0]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        
        # Generar mapa de calor
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= np.max(heatmap)  # normalizar
        
        # Redimensionar y aplicar colormap
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superponer heatmap en la imagen original
        if len(original_image.shape) == 2:  # Si es imagen en escala de grises
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        alpha = 0.5  # Factor de mezcla
        superimposed = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
        
        return superimposed[:, :, ::-1]  # Convertir BGR a RGB