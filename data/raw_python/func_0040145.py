def source_to_unicode(txt, errors='replace', skip_encoding_cookie=True):
    """Converts a bytes string with python source code to unicode.
    Unicode strings are passed through unchanged. Byte strings are checked
    for the python source file encoding cookie to determine encoding.
    txt can be either a bytes buffer or a string containing the source
    code.
    """
    if isinstance(txt, six.text_type):
        return txt
    if isinstance(txt, six.binary_type):
        buffer = io.BytesIO(txt)
    else:
        buffer = txt
    try:
        encoding, _ = detect_encoding(buffer.readline)
    except SyntaxError:
        encoding = "ascii"
    buffer.seek(0)

    newline_decoder = io.IncrementalNewlineDecoder(None, True)

    text = io.TextIOWrapper(buffer, encoding, errors=errors, line_buffering=True)
    text.mode = 'r'
    if skip_encoding_cookie:
        return u"".join(strip_encoding_cookie(text))
    else:
        return text.read()