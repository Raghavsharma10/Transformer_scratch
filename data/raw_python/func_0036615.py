def configuration(self):
        '''The configuration dictionary of the manager.'''
        with self._mutex:
            if not self._configuration:
                self._configuration = utils.nvlist_to_dict(self._obj.get_configuration())
        return self._configuration