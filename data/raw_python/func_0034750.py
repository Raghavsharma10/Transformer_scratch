def calculate_partial_digest(username, realm, password):
    '''
    Calculate a partial digest that may be stored and used to authenticate future
    HTTP Digest sessions.
    '''
    return md5.md5("%s:%s:%s" % (username.encode('utf-8'), realm, password.encode('utf-8'))).hexdigest()