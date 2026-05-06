def getHashForFile(f):
    """Returns a hash value for a file

    :param f: File to hash
    :type f: str
    :returns: str
    """
    hashVal = hashlib.sha1()
    while True:
        r = f.read(1024)
        if not r:
            break
        hashVal.update(r)
    f.seek(0)

    return hashVal.hexdigest()