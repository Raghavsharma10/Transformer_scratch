def ripesha(msg):
    """RIPEMD160(SHA256(msg)) -> HASH object"""
    ripe = hashlib.new('ripemd160')
    ripe.update(hashlib.sha256(msg).digest())
    return ripe