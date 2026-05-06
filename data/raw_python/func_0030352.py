def encrypt_password(raw_password, algorithm='sha1', salt=None):
    """
    Returns a string of the hexdigest of the given plaintext password and salt
    using the given algorithm ('md5', 'sha1' or other supported by hashlib).
    """
    if salt is None:
        salt = binascii.hexlify(os.urandom(3))[:5]
    else:
        salt = salt.encode('utf-8')

    raw_password = raw_password.encode('utf-8')
    hash = hashlib.new(algorithm, salt+raw_password).hexdigest()
    return '{}${}${}'.format(algorithm, salt.decode('utf-8'), hash)