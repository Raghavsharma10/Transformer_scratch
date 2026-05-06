def _safe_get(mapping, key, default=None):
    """Helper for accessing style values.

    It exists to avoid checking whether `mapping` is indeed a mapping before
    trying to get a key.  In the context of style dicts, this eliminates "is
    this a mapping" checks in two common situations: 1) a style argument is
    None, and 2) a style key's value (e.g., width) can be either a mapping or a
    plain value.
    """
    try:
        return mapping.get(key, default)
    except AttributeError:
        return default