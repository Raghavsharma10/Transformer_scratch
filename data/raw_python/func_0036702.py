def active_conf_set(self):
        '''The currently-active configuration set.'''
        with self._mutex:
            if not self.conf_sets:
                return None
            if not self._active_conf_set:
                return None
            return self.conf_sets[self._active_conf_set]