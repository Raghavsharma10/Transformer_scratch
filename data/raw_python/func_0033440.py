def lookup(self, fact, cache=True):
        """Return the value of a given fact and raise a KeyError if
        it is not available. If `cache` is False, force the lookup of
        the fact."""
        if (not cache) or (not self.has_cache()):
            val =  self.run_facter(fact)
            if val is None or val == '':
                raise KeyError(fact)
            return val
        return self._cache[fact]