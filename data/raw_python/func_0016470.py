def raw_encode(data):
    """Special case serializer."""
    content_type = 'application/data'
    payload = data
    if isinstance(payload, unicode):
        content_encoding = 'utf-8'
        payload = payload.encode(content_encoding)
    else:
        content_encoding = 'binary'
    return content_type, content_encoding, payload