def randkey(bits, keyspace=string.ascii_letters + string.digits + '#/.',
            rng=None):
    """ Returns a cryptographically secure random key of desired @bits of
        entropy within @keyspace using :class:random.SystemRandom

        @bits: (#int) minimum bits of entropy
        @keyspace: (#str) or iterable allowed output chars
        @rng: the random number generator to use. Defaults to
            :class:random.SystemRandom. Must have a |choice| method

        -> (#str) random key

        ..
            from vital.security import randkey

            randkey(24)
            # -> '9qaX'
            randkey(48)
            # -> 'iPJ5YWs9'
            randkey(64)
            # - > 'C..VJ.KLdxg'
            randkey(64, keyspace="abc", rng=random)
            # -> 'aabcccbabcaacaccccabcaabbabcacabacbbbaaab'
        ..
    """
    return "".join(char for char in iter_random_chars(bits, keyspace, rng))