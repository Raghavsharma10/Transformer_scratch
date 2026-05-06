def compute_sha256(path):
    """
    Compute the SHA-256 hash of the file at the given path

    Parameters
    ----------
    path: str
        The path of the file

    Returns
    -------
    str
        The SHA-256 HEX digest
    """
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        # 10MB chunks
        for chunk in iter(lambda: f.read(10 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()