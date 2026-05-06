def build_from_features(self, features):
        """
        :param features: a list of (token,weight) tuples or a token -> weight dict,
                        if is a string so it need compute weight (a weight of 1 will be assumed).

        :return: a decimal digit for the accumulative result of each after handled features-weight pair.
        """
        v = [0] * self.hash_bit_number
        if isinstance(features, dict):
            features = features.items()

        # Starting longitudinal accumulation of bits, current bit add current weight
        # when the current bits equal 1 and else current bit minus the current weight.
        for f in features:
            if isinstance(f, str):
                h = self.hashfunc(f, self.hash_bit_number)
                w = 1
            else:
                assert isinstance(f, collections.Iterable)
                h = self.hashfunc(f[0], self.hash_bit_number)
                w = f[1]
            for i in range(self.hash_bit_number):
                bitmask = 1 << i
                v[i] += w if h & bitmask else -w

        # Just record weight of the non-negative
        fingerprint = 0
        for i in range(self.hash_bit_number):
            if v[i] >= 0:
                fingerprint += 1 << i

        return fingerprint