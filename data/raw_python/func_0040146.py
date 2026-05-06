def decode_source(source_bytes):
    """Decode bytes representing source code and return the string.
    Universal newline support is used in the decoding.
    """
    # source_bytes_readline = io.BytesIO(source_bytes).readline
    # encoding, _ = detect_encoding(source_bytes_readline)
    newline_decoder = io.IncrementalNewlineDecoder(None, True)
    return newline_decoder.decode(source_to_unicode(source_bytes))