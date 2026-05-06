def _add_new_items(self, config, seen):
        '''Add new (unseen) items to the config.'''
        for (key, value) in self.items():
            if key not in seen:
                self._set_value(config, key, value)