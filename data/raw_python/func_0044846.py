def loadb(b):
    """Deserialize ``b`` (instance of ``bytes``) to a Python object."""
    assert isinstance(b, (bytes, bytearray))
    return std_json.loads(b.decode('utf-8'))