def random_string(length=6, alphabet=string.ascii_letters+string.digits):
    """
    Return a random string of given length and alphabet.

    Default alphabet is url-friendly (base62).
    """
    return ''.join([random.choice(alphabet) for i in xrange(length)])