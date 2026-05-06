def CA(self):
        '''
        Vertices C and A, list.

        '''
        try:
            return self._CA
        except AttributeError:
            pass
        self._CA = [self.C, self.A]
        return self._CA