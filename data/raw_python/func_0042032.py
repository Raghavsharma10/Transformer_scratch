def _sync_and_resolve(self, config, resolver):
        '''Synchronize all items represented by the config according to the resolver and return a
        set of keys that have been resolved.'''
        resolved = set()
        for key, theirs in config.items(self._name):
            theirs = self._real_value_of(theirs)
            if key in self:
                mine = self[key]
                value = resolver(self._name, key, mine, theirs)
            else:
                value = theirs
            self._set_value(config, key, value)
            resolved.add(key)
        return resolved