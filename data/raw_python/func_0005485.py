def randstr(size, keyspace=string.ascii_letters + string.digits, rng=None):
    """ Returns a cryptographically secure random string of desired @size
        (in character length) within @keyspace using :class:random.SystemRandom

        @size: (#int) number of random characters to generate
        @keyspace: (#str) or iterable allowed output chars
        @rng: the random number generator to use. Defaults to
            :class:random.SystemRandom. Must have a |choice| method

        -> #str random key

        ..
            from vital.security import randkey

            randstr(4)
            # -> '9qaX'
        ..
    """
    rng = rng or random.SystemRandom()
    return "".join(rng.choice(keyspace) for char in range(int(ceil(size))))