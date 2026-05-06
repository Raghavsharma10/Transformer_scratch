def CB(self):
        '''
        Vertices C and B, list.

        '''
        try:
            return self._CB
        except AttributeError:
            pass
        self._CB = [self.C, self.B]
        return self._CB