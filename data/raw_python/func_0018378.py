def hash_of_file(path):
    """Return the hash of a downloaded file."""
    with open(path, 'rb') as archive:
        sha = sha256()
        while True:
            data = archive.read(2 ** 20)
            if not data:
                break
            sha.update(data)
    return encoded_hash(sha)