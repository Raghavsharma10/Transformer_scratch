def _direct_set(self, key, value):
        '''
            _direct_set - INTERNAL USE ONLY!!!!

                Directly sets a value on the underlying dict, without running through the setitem logic

        '''
        dict.__setitem__(self, key, value)
        return value