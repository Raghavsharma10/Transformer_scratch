def random_str(Nchars=6, randstrbase='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    """Return a random string of <Nchars> characters. Characters are sampled
    uniformly from <randstrbase>.
    """
    return ''.join([randstrbase[random.randint(0, len(randstrbase) - 1)] for i in range(Nchars)])