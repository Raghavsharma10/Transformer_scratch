def _read_message(self):
    """ Reads a single size-annotated message from the server """
    size = int(self.buf.read_line().decode("utf-8"))
    return self.buf.read(size).decode("utf-8")