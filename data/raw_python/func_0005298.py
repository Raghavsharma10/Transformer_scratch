def open_file(filename, as_text=False):
    """Open the file gunzipping it if it ends with .gz.
    If as_text the file is opened in text mode,
    otherwise the file's opened in binary mode."""
    if filename.lower().endswith('.gz'):
        if as_text:
            return gzip.open(filename, 'rt')
        else:
            return gzip.open(filename, 'rb')
    else:
        if as_text:
            return open(filename, 'rt')
        else:
            return open(filename, 'rb')