def data(self, data=None, ret_r=False):
        '''response data'''
        if data or ret_r:
            self._data = data
            return self
        return self._data