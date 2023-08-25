import io


def save_to_buffer(func) -> bytes:
    buffer = io.BytesIO()
    func(buffer)
    return buffer.getvalue()
