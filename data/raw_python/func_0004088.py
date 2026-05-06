def guess_path_encoding(file_path, default=DEFAULT_ENCODING):
    """Wrapper to open that damn file for you, lazy bastard."""
    with io.open(file_path, 'rb') as fh:
        return guess_file_encoding(fh, default=default)