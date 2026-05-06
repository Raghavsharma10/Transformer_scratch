def fromdict(dict):
        """Takes a dictionary as an argument and returns a new Proof object
        from the dictionary.

        :param dict: the dictionary to convert
        """
        self = Proof()
        self.mu = dict["mu"]
        self.sigma = dict["sigma"]
        return self