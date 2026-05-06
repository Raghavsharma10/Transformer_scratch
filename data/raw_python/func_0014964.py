def _get_random_id():
    """ Get a random (i.e., unique) string identifier"""
    symbols = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(symbols) for _ in range(15))