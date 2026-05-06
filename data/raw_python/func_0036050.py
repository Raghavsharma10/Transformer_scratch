def fromdict(dict):
        """Takes a dictionary as an argument and returns a new Challenge
        object from the dictionary.

        :param dict: the dictionary to convert
        """
        seed = hb_decode(dict['seed'])
        index = dict['index']
        return Challenge(seed, index)