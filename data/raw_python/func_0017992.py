def make_tempfile(data=None):
    "Create a temp file, write our PID into it."
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
        temp.write(six.text_type(data if data is not None else os.getpid()))
        return temp.name