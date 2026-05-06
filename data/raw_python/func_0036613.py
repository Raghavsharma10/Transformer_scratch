def factory_profiles(self):
        '''The factory profiles of all loaded modules.'''
        with self._mutex:
            if not self._factory_profiles:
                self._factory_profiles = []
                for fp in self._obj.get_factory_profiles():
                    self._factory_profiles.append(utils.nvlist_to_dict(fp.properties))
        return self._factory_profiles