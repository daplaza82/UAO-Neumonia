# Usar una imagen base con Python 3.11.4
FROM python:3.11.4-slim

# Establecer variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DISPLAY=host.docker.internal:0.0

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema incluyendo Tkinter y X11
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    python3-tk \
    tk-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y recursos
COPY src/ ./src/
COPY models/ ./models/
COPY test_images/ ./test_images/

# Crear un usuario no root y dar permisos
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Variables de entorno adicionales
ENV PYTHONPATH=/app

# Puerto para la interfaz gráfica
EXPOSE 8080

# Configurar el punto de entrada
ENTRYPOINT ["python"]
CMD ["src/gui/app.py"]

# Metadata
LABEL maintainer="David Plaza C <daplaza82@gmail.com>" \
      version="1.0" \
      description="Detector de Neumonía usando IA"