def b58enc(uid):
    '''Encodes a UID to an 11-length string, encoded using base58 url-safe alphabet'''
    # note: i tested a buffer array too, but string concat was 2x faster
    if not isinstance(uid, int):
        raise ValueError('Invalid integer: {}'.format(uid))
    if uid == 0:
        return BASE58CHARS[0]
    enc_uid = ""
    while uid:
        uid, r = divmod(uid, 58)
        enc_uid = BASE58CHARS[r] + enc_uid
    return enc_uid