def resolve(self, key, keylist):
        """Hook to resolve ambiguities in selected keys"""
        raise AmbiguousKeyError("Ambiguous key "+ repr(key) +
                ", could be any of " + str(sorted(keylist)))