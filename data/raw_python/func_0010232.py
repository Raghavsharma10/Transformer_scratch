def BA(self):
        '''
        Vertices B and A, list.

        '''
        try:
            return self._BA
        except AttributeError:
            pass
        self._BA = [self.B, self.A]
        return self._BA