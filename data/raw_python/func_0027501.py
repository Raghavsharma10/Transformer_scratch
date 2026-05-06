def gen_secret(length=64):
    """ Generates a secret of given length
    """
    charset = string.ascii_letters + string.digits
    return ''.join(random.SystemRandom().choice(charset)
                   for _ in range(length))