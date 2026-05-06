def random_hex(length):
    """Generates a random hex string"""
    return escape.to_unicode(binascii.hexlify(os.urandom(length))[length:])