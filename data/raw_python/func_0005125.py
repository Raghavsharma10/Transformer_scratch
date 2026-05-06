def gen_rand_str(*size, use=None, keyspace=None):
    """ Generates a random string using random module specified in @use within
        the @keyspace

        @*size: #int size range for the length of the string
        @use: the random module to use
        @keyspace: #str chars allowed in the random string
        ..
            from vital.debug import gen_rand_str

            gen_rand_str()
            # -> 'PRCpAq'

            gen_rand_str(1, 2)
            # -> 'Y'

            gen_rand_str(12, keyspace="abcdefg")
            # -> 'gaaacffbedf'
        ..
    """
    keyspace = keyspace or (string.ascii_letters + string.digits)
    keyspace = [char for char in keyspace]
    use = use or _random
    use.seed()
    if size:
        size = size if len(size) == 2 else (size[0], size[0])
    else:
        size = (10, 12)
    return ''.join(
        use.choice(keyspace)
        for _ in range(use.randint(*size)))