def normalize_result(result, default, threshold=0.2):
    """Interpret a chardet result."""
    if result is None:
        return default
    if result.get('confidence') is None:
        return default
    if result.get('confidence') < threshold:
        return default
    return normalize_encoding(result.get('encoding'),
                              default=default)