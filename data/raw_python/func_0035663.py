def generate_seeds(num, root_seed, secret):
        """ Deterministically generate list of seeds from a root seed.

        :param num: Numbers of seeds to generate as int
        :param root_seed: Seed to start off with.
        :return: seed values as a list of length num
        """
        # Generate a starting seed from the root
        if num < 0:
            raise HeartbeatError('%s is not greater than 0' % num)

        if secret is None:
            raise HeartbeatError('secret can not be of type NoneType')

        seeds = []
        try:
            tmp_seed = hashlib.sha256(root_seed).digest()
        except TypeError:
            tmp_seed = hashlib.sha256(str(root_seed).encode()).digest()

        # Deterministically generate the rest of the seeds
        for x in range(num):
            seeds.append(tmp_seed)
            h = hashlib.sha256(tmp_seed)
            h.update(secret)
            tmp_seed = h.digest()

        return seeds