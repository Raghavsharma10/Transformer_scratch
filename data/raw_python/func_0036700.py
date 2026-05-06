def activate_conf_set(self, set_name):
        '''Activate a configuration set by name.

        @raises NoSuchConfSetError

        '''
        with self._mutex:
            if not set_name in self.conf_sets:
                raise exceptions.NoSuchConfSetError(set_name)
            self._conf.activate_configuration_set(set_name)