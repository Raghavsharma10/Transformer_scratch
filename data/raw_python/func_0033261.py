def find_path_with_profiles(self, conversion_profiles, in_, out):
        '''
        Like find_path, except forces the conversion profiles to be the given
        conversion profile setting. Useful for "temporarily overriding" the
        global conversion profiles with your own.
        '''
        original_profiles = dict(self.conversion_profiles)
        self._setup_profiles(conversion_profiles)
        results = self.find_path(in_, out)
        self.conversion_profiles = original_profiles
        return results