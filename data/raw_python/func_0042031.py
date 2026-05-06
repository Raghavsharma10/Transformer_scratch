def sync_with(self, config, conflict_resolver):
        '''Synchronizes current set of key/values in this instance with those in the config.'''
        if not config.has_section(self._name):
            config.add_section(self._name)
        resolved = self._sync_and_resolve(config, conflict_resolver)
        self._add_new_items(config, resolved)