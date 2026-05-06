def normalize_encoding(encoding, default=DEFAULT_ENCODING):
    """Normalize the encoding name, replace ASCII w/ UTF-8."""
    if encoding is None:
        return default
    encoding = encoding.lower().strip()
    if encoding in ['', 'ascii']:
        return default
    try:
        codecs.lookup(encoding)
        return encoding
    except LookupError:
        return default