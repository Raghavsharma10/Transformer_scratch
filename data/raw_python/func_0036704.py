def conf_sets(self):
        '''The dictionary of configuration sets in this component, if any.'''
        with self._mutex:
            if not self._conf_sets:
                self._parse_configuration()
        return self._conf_sets