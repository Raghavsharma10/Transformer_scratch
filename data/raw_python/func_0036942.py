def _get_crypto(keylen, hexkey, key):
    """Return a camcrypt.CamCrypt object based on keylen, hexkey, and key."""
    if keylen not in camcrypt.ACCEPTABLE_KEY_LENGTHS:
        raise ValueError("key length must be one of 128, 192, or 256")

    if hexkey:
        key = key.decode('hex')

    return camcrypt.CamCrypt(keylen=keylen, key=key)