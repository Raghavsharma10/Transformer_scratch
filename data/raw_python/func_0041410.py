def shasha(msg):
    """SHA256(SHA256(msg)) -> HASH object"""
    res = hashlib.sha256(hashlib.sha256(msg).digest())
    return res