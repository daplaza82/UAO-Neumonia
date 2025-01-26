"""
Módulo principal de la interfaz gráfica para la detección de neumonía.
"""

import tkinter as tk
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
from pathlib import Path
from abc import ABC, abstractmethod
import csv
from PIL import ImageTk, Image
import tkcap

from src.integrator import PneumoniaDetector
from src.read_img import ImageReaderFactory  # Añadimos esta importación


class UIComponent(ABC):
    """Clase abstracta base para componentes de la interfaz."""
    
    @abstractmethod
    def create_widgets(self):
        """Crea los widgets del componente."""
        pass
        
    @abstractmethod
    def place_widgets(self):
        """Posiciona los widgets en la ventana."""
        pass


class ImageDisplay(UIComponent):
    """Componente para mostrar imágenes."""
    
    def __init__(self, master, title: str, x: int, y: int):
        """
        Inicializa el componente de visualización de imagen.
        
        Args:
            master: Widget padre
            title (str): Título del componente
            x (int): Posición x
            y (int): Posición y
        """
        self.master = master
        self.title = title
        self.x = x
        self.y = y
        self.image = None
        self.create_widgets()
        self.place_widgets()
    
    def create_widgets(self):
        """Crea los widgets necesarios para mostrar la imagen."""
        self.label = ttk.Label(self.master, text=self.title, 
                             font=font.Font(weight="bold"))
        self.display = tk.Text(self.master, width=31, height=15)
    
    def place_widgets(self):
        """Posiciona los widgets en la ventana."""
        self.label.place(x=self.x, y=self.y)
        self.display.place(x=self.x-45, y=self.y+25)
    
    def show_image(self, image):
        """
        Muestra una imagen en el componente.
        
        Args:
            image: Imagen a mostrar (PIL Image o numpy array)
        """
        self.image = image
        if isinstance(image, ImageTk.PhotoImage):
            self.display.image_create(tk.END, image=image)
        else:
            # Convertir y mostrar la imagen
            img = Image.fromarray(image)
            img = img.resize((250, 250), Image.Resampling.LANCZOS)
            self.image = ImageTk.PhotoImage(img)
            self.display.image_create(tk.END, image=self.image)
    
    def clear(self):
        """Limpia la imagen mostrada."""
        if self.image:
            self.display.delete("1.0", tk.END)


class ResultDisplay(UIComponent):
    """Componente para mostrar resultados."""
    
    def __init__(self, master, x: int, y: int):
        """
        Inicializa el componente de visualización de resultados.
        
        Args:
            master: Widget padre
            x (int): Posición x
            y (int): Posición y
        """
        self.master = master
        self.x = x
        self.y = y
        self.create_widgets()
        self.place_widgets()
    
    def create_widgets(self):
        """Crea los widgets para mostrar los resultados."""
        self.result_label = ttk.Label(self.master, text="Resultado:", 
                                    font=font.Font(weight="bold"))
        self.prob_label = ttk.Label(self.master, text="Probabilidad:", 
                                  font=font.Font(weight="bold"))
        self.result_text = tk.Text(self.master, width=10, height=1)
        self.prob_text = tk.Text(self.master, width=10, height=1)
    
    def place_widgets(self):
        """Posiciona los widgets en la ventana."""
        self.result_label.place(x=self.x, y=self.y)
        self.prob_label.place(x=self.x, y=self.y+50)
        self.result_text.place(x=self.x+110, y=self.y)
        self.prob_text.place(x=self.x+110, y=self.y+50)
    
    def show_results(self, prediction: str, probability: float):
        """
        Muestra los resultados de la predicción.
        
        Args:
            prediction (str): Clase predicha
            probability (float): Probabilidad de la predicción
        """
        self.result_text.delete("1.0", tk.END)
        self.prob_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, prediction)
        self.prob_text.insert(tk.END, f"{probability:.2f}%")
    
    def clear(self):
        """Limpia los resultados mostrados."""
        self.result_text.delete("1.0", tk.END)
        self.prob_text.delete("1.0", tk.END)


class PneumoniaDetectorGUI:
    """Clase principal de la interfaz gráfica."""
    
    def __init__(self):
        """Inicializa la aplicación GUI."""
        self.root = tk.Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)
        
        # Inicializar el detector
        self.detector = PneumoniaDetector()
        
        # Inicializar componentes
        self._create_components()
        self._create_buttons()
        
        # Variables de estado
        self.current_image_path = None
        self.report_id = 0
    
    def _create_components(self):
        """Crea los componentes principales de la interfaz."""
        # Título principal
        self.title_label = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=font.Font(weight="bold")
        )
        self.title_label.place(x=122, y=25)
        
        # Componentes de visualización
        self.original_display = ImageDisplay(self.root, "Imagen Radiográfica", 155, 65)
        self.heatmap_display = ImageDisplay(self.root, "Imagen con Heatmap", 590, 65)
        
        # Componente de resultados
        self.result_display = ResultDisplay(self.root, 500, 350)
        
        # Campo de ID del paciente
        self.id_label = ttk.Label(self.root, text="Cédula Paciente:", 
                                font=font.Font(weight="bold"))
        self.id_label.place(x=65, y=350)
        self.id_entry = ttk.Entry(self.root, width=10)
        self.id_entry.place(x=200, y=350)
    
    def _create_buttons(self):
        """Crea los botones de la interfaz."""
        self.load_button = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.predict_button = ttk.Button(
            self.root, text="Predecir", command=self._predict, state="disabled"
        )
        self.save_button = ttk.Button(
            self.root, text="Guardar", command=self._save_results
        )
        self.pdf_button = ttk.Button(
            self.root, text="PDF", command=self._create_pdf
        )
        self.clear_button = ttk.Button(
            self.root, text="Borrar", command=self._clear_all
        )
        
        # Posicionar botones
        self.load_button.place(x=70, y=460)
        self.predict_button.place(x=220, y=460)
        self.save_button.place(x=370, y=460)
        self.pdf_button.place(x=520, y=460)
        self.clear_button.place(x=670, y=460)
    
    def load_img_file(self):
        """Maneja la carga de una imagen."""
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Seleccionar imagen",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:
            try:
                # Obtener la extensión del archivo
                file_extension = Path(filepath).suffix[1:]
                
                # Usar el factory para obtener el lector apropiado
                reader = ImageReaderFactory.get_reader(file_extension)
                self.array, img2show = reader.read(filepath)
                
                # Limpiar imagen anterior si existe
                self.original_display.clear()
                
                # Redimensionar y mostrar la nueva imagen
                self.img1 = img2show.resize((250, 250), Image.Resampling.LANCZOS)
                self.img1 = ImageTk.PhotoImage(self.img1)
                self.original_display.show_image(self.img1)
                
                # Habilitar botón de predicción
                self.predict_button["state"] = "enabled"
                
            except Exception as e:
                showinfo("Error", f"Error al cargar la imagen: {str(e)}")
                self.predict_button["state"] = "disabled"
    
    def _predict(self):
        """Realiza la predicción usando el detector."""
        try:
            if hasattr(self, 'array') and self.array is not None:
                # Usar el array de la imagen cargada
                label, prob, heatmap = self.detector.process_image(self.array)
                
                # Mostrar resultados
                self.result_display.show_results(label, prob)
                
                # Limpiar heatmap anterior si existe
                self.heatmap_display.clear()
                
                # Mostrar nuevo heatmap
                self.heatmap_display.show_image(heatmap)
            else:
                showinfo("Error", "Por favor cargue una imagen primero.")
        except Exception as e:
            import traceback
            print(f"Error detallado: {traceback.format_exc()}")  # Para depuración
            showinfo("Error", f"Error al procesar la imagen: {str(e)}")
    
    def _save_results(self):
        """Guarda los resultados en un archivo CSV."""
        if not self.id_entry.get():
            showinfo("Error", "Por favor ingrese la cédula del paciente.")
            return
            
        try:
            with open("historial.csv", "a", newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter="-")
                writer.writerow([
                    self.id_entry.get(),
                    self.result_display.result_text.get("1.0", tk.END).strip(),
                    self.result_display.prob_text.get("1.0", tk.END).strip()
                ])
            showinfo("Guardar", "Los datos se guardaron con éxito.")
        except Exception as e:
            showinfo("Error", f"Error al guardar los datos: {str(e)}")
    
    def _create_pdf(self):
        """Genera un PDF del reporte actual."""
        try:
            cap = tkcap.CAP(self.root)
            img_path = f"Reporte{self.report_id}.jpg"
            cap.capture(img_path)
            
            img = Image.open(img_path)
            img = img.convert("RGB")
            pdf_path = f"Reporte{self.report_id}.pdf"
            img.save(pdf_path)
            
            self.report_id += 1
            showinfo("PDF", "El PDF fue generado con éxito.")
            
        except Exception as e:
            showinfo("Error", f"Error al generar el PDF: {str(e)}")
    
    def _clear_all(self):
        """Limpia todos los campos y visualizaciones."""
        if askokcancel("Confirmación", "Se borrarán todos los datos.", 
                      icon=WARNING):
            self.original_display.clear()
            self.heatmap_display.clear()
            self.result_display.clear()
            self.id_entry.delete(0, tk.END)
            self.array = None
            self.predict_button["state"] = "disabled"
            showinfo("Borrar", "Los datos se borraron con éxito")
    
    def run(self):
        """Inicia la aplicación."""
        self.root.mainloop()


def main():
    """Función principal para iniciar la aplicación."""
    app = PneumoniaDetectorGUI()
    app.run()


if __name__ == "__main__":
    main()