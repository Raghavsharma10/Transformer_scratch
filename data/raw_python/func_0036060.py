def get_next_seed(key, seed):
        """This takes a seed and generates the next seed in the sequence.
        it simply calculates the hmac of the seed with the key.  It returns
        the next seed

        :param key: the key to use for the HMAC
        :param seed: the seed to permutate
        """
        return hmac.new(key, seed, hashlib.sha256).digest()