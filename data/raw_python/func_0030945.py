def get_file_md5sum(path):
    """Calculate the MD5 hash for a file."""
    with open(path, 'rb') as fh:
        h = str(hashlib.md5(fh.read()).hexdigest())
    return h