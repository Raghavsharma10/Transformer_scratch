def sha1sum(f):
    """Return the SHA-1 hash of the contents of file `f`, in hex format"""
    h = hashlib.sha1()
    fp = open(f, 'rb')
    while True:
        block = fp.read(512 * 1024)
        if not block:
            break
        h.update(block)
    return h.hexdigest()