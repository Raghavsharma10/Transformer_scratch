def _serialize_buffer(buffer, array_serialization=None):
    """Serialize a NumPy array."""
    if array_serialization == 'binary':
        # WARNING: in NumPy 1.9, tostring() has been renamed to tobytes()
        # but tostring() is still here for now for backward compatibility.
        return buffer.ravel().tostring()
    elif array_serialization == 'base64':
        return {'storage_type': 'base64',
                'buffer': base64.b64encode(buffer).decode('ascii')
                }
    raise ValueError("The array serialization method should be 'binary' or "
                     "'base64'.")