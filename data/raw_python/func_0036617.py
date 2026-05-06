def loadable_modules(self):
        '''The list of loadable module profile dictionaries.'''
        with self._mutex:
            if not self._loadable_modules:
                self._loadable_modules = []
                for mp in self._obj.get_loadable_modules():
                    self._loadable_modules.append(utils.nvlist_to_dict(mp.properties))
        return self._loadable_modules