def profile(self):
        '''The manager's profile.'''
        with self._mutex:
            if not self._profile:
                profile = self._obj.get_profile()
                self._profile = utils.nvlist_to_dict(profile.properties)
        return self._profile