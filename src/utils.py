"""Утилиты"""

import io


def save_to_buffer(func) -> bytes:
    """Сохранение в буффер"""
    buffer = io.BytesIO()
    func(buffer)
    return buffer.getvalue()
