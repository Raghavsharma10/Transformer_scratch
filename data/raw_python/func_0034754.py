def calculate_nonce(timestamp, secret, salt=None):
    '''
    Generate a nonce using the provided timestamp, secret, and salt. If the salt is not provided,
    (and one should only be provided when validating a nonce) one will be generated randomly
    in order to ensure that two simultaneous requests do not generate identical nonces.
    '''
    if not salt:
        salt = ''.join([random.choice('0123456789ABCDEF') for x in range(4)])
    return "%s:%s:%s" % (timestamp, salt,
                         md5.md5("%s:%s:%s" % (timestamp, salt, secret)).hexdigest())