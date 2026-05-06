def _setup_profiles(self, conversion_profiles):
        '''
        Add given conversion profiles checking for invalid profiles
        '''
        # Check for invalid profiles
        for key, path in conversion_profiles.items():
            if isinstance(path, str):
                path = (path, )
            for left, right in pair_looper(path):
                pair = (_format(left), _format(right))
                if pair not in self.converters:
                    msg = 'Invalid conversion profile %s, unknown step %s'
                    log.warning(msg % (repr(key), repr(pair)))
                    break
            else:
                # If it did not break, then add to conversion profiles
                self.conversion_profiles[key] = path