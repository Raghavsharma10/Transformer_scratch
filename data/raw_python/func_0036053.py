def fromdict(dict):
        """Takes a dictionary as an argument and returns a new State object
        from the dictionary.

        :param dict: the dictionary to convert
        """
        index = dict['index']
        seed = hb_decode(dict['seed'])
        n = dict['n']
        root = hb_decode(dict['root'])
        hmac = hb_decode(dict['hmac'])
        timestamp = dict['timestamp']
        self = State(index, seed, n, root, hmac, timestamp)
        return self