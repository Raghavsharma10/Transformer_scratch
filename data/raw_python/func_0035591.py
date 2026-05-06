def findInvariant(self, symclasses):
        """(symclasses) -> converge the disambiguity function
        until we have an invariant"""
        get = primes.primes.get
        disambiguate = self.disambiguate
        breakRankTies = self.breakRankTies

        while 1:
            newSyms = map(get, symclasses)
            newSyms = disambiguate(newSyms)
            breakRankTies(symclasses, newSyms)
            if symclasses == newSyms:
                return newSyms
            symclasses = newSyms