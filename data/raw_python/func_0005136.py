def randdomain(self):
        """ -> a randomized domain-like name """
        return '.'.join(
            rand_readable(3, 6, use=self.random, density=3)
            for _ in range(self.random.randint(1, 2))
        ).lower()