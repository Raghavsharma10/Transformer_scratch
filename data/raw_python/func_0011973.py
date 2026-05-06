def _isclose(obja, objb, rtol=1e-05, atol=1e-08):
    """Return floating point equality."""
    return abs(obja - objb) <= (atol + rtol * abs(objb))