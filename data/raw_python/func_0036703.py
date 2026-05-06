def active_conf_set_name(self):
        '''The name of the currently-active configuration set.'''
        with self._mutex:
            if not self.conf_sets:
                return ''
            if not self._active_conf_set:
                return ''
            return self._active_conf_set