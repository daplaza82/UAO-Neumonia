"""
Tests para el módulo de lectura de imágenes.
"""

import pytest
from src.read_img import ImageReaderFactory, DicomReader, JpgReader


def test_factory_returns_correct_reader():
    """Prueba que la fábrica retorne el lector correcto según la extensión."""
    factory = ImageReaderFactory()
    
    # Probar con diferentes extensiones
    assert isinstance(factory.get_reader('dcm'), DicomReader)
    assert isinstance(factory.get_reader('jpg'), JpgReader)
    assert isinstance(factory.get_reader('jpeg'), JpgReader)
    assert isinstance(factory.get_reader('png'), JpgReader)


def test_factory_raises_error_for_unsupported_format():
    """Prueba que se lance un error para formatos no soportados."""
    factory = ImageReaderFactory()
    
    with pytest.raises(ValueError):
        factory.get_reader('unsupported')