def _read(path, encoding="utf-8", comment=";;;"):
    """ Returns an iterator over the lines in the file at the given path,
        strippping comments and decoding each line to Unicode.
    """
    if path:
        if isinstance(path, basestring) and os.path.exists(path):
            # From file path.
            if PY2:
                f = codecs.open(path, 'r', encoding='utf-8')
            else:
                f = open(path, 'r', encoding='utf-8')
        elif isinstance(path, basestring):
            # From string.
            f = path.splitlines()
        else:
            # From file or buffer.
            f = path
        for i, line in enumerate(f):
            line = line.strip(codecs.BOM_UTF8) if i == 0 and isinstance(line, binary_type) else line
            line = line.strip()
            line = decode_utf8(line, encoding)
            if not line or (comment and line.startswith(comment)):
                continue
            yield line
    return