def hash_str(data, hasher=None):
    """Checksum hash a string."""
    hasher = hasher or hashlib.sha1()
    hasher.update(data)
    return hasher