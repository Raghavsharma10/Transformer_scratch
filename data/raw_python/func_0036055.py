def fromdict(dict):
        """Takes a dictionary as an argument and returns a new Proof object
        from the dictionary.

        :param dict: the dictionary to convert
        """
        key = hb_decode(dict['key'])
        check_fraction = dict['check_fraction']
        return Merkle(check_fraction, key)