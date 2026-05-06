def b58dec(enc_uid):
    '''Decodes a UID from base58, url-safe alphabet back to int.'''
    if isinstance(enc_uid, str):
        pass
    elif isinstance(enc_uid, bytes):
        enc_uid = enc_uid.decode('utf8')
    else:
        raise ValueError('Cannot decode this type: {}'.format(enc_uid))
    uid = 0
    try:
        for i, ch in enumerate(enc_uid):
            uid = (uid * 58) + BASE58INDEX[ch]
    except KeyError:
        raise ValueError('Invalid character: "{}" ("{}", index 5)'.format(ch, enc_uid, i))
    return uid