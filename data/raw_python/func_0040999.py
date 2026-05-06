def get_hash(self):
        """Retruns a hash based on the the current table code and kwargs.
        Also changes based on dependent tables."""
        depencency_hashes = [dep.get_hash() for dep in self.dep()]
        sl = inspect.getsourcelines
        hash_sources = [sl(self.__class__), self.args,
                        self.kwargs, *depencency_hashes]
        hash_input = pickle.dumps(hash_sources)
        return hashlib.md5(hash_input).hexdigest()