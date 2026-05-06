def aes_pad_key(key):
    """
    AES keys must be either 16, 24, or 32 bytes long. If a key is provided that is not
    one of these lengths, pad it with zeroes (this is what pgcrypto does).
    """
    if len(key) in (16, 24, 32):
        return key
    if len(key) < 16:
        return pad(key, 16, zero=True)
    elif len(key) < 24:
        return pad(key, 24, zero=True)
    else:
        return pad(key[:32], 32, zero=True)