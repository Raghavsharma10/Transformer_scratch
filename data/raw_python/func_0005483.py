def iter_random_chars(bits,
                      keyspace=string.ascii_letters + string.digits + '#/.',
                      rng=None):
    """ Yields a cryptographically secure random key of desired @bits of
        entropy within @keyspace using :class:random.SystemRandom

        @bits: (#int) minimum bits of entropy
        @keyspace: (#str) or iterable allowed output chars

        ..
            from vital.security import iter_rand

            for char in iter_rand(512):
                do_something_with(char)
    """
    if bits < 8:
        raise ValueError('Bits cannot be <8')
    else:
        chars = chars_in(bits, keyspace)
    rng = rng or random.SystemRandom()
    for char in range(int(ceil(chars))):
        yield rng.choice(keyspace)