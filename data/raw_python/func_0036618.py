def loaded_modules(self):
        '''The list of loaded module profile dictionaries.'''
        with self._mutex:
            if not self._loaded_modules:
                self._loaded_modules = []
                for mp in self._obj.get_loaded_modules():
                    self._loaded_modules.append(utils.nvlist_to_dict(mp.properties))
        return self._loaded_modules