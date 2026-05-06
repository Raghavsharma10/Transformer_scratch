def a_temp_file():
    """Yield the name of a temporary file and ensure it's removed after use"""
    filename = None
    try:
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        filename = tmpfile.name
        yield tmpfile
    finally:
        if filename and os.path.exists(filename):
            os.remove(filename)