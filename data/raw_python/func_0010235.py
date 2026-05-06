def BC(self):
        '''
        Vertices B and C, list.

        '''
        try:
            return self._BC
        except AttributeError:
            pass
        self._BC = [self.B, self.C]
        return self._BC