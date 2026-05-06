def string_to_bytes(string, size):
    """Convert string to bytes add padding."""
    if len(string) > size:
        raise PyVLXException("string_to_bytes::string_to_large")
    encoded = bytes(string, encoding='utf-8')
    return encoded + bytes(size-len(encoded))