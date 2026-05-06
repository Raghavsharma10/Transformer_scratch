def hash_text(text):
    """Return MD5 hash of a string."""
    md5 = hashlib.md5()
    md5.update(text)
    return md5.hexdigest()