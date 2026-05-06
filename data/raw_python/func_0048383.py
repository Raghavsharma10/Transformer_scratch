def hexlify_pbkdf2(password, salt, iterations, length, digest=hashlib.sha1):
    """
    >>> hash = hexlify_pbkdf2("not secret", "a salt value", iterations=100, length=16)
    >>> hash == '0b919231515dde16f76364666cf07107'
    True
    """
    # log.debug("hexlify_pbkdf2 with iterations=%i", iterations)
    hash = crypto.pbkdf2(password, salt, iterations=iterations, dklen=length, digest=digest)
    hash = binascii.hexlify(hash)
    hash = six.text_type(hash, "ascii")
    return hash