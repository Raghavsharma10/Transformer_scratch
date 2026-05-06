def AB(self):
        '''
        A list containing Points A and B.
        '''
        try:
            return self._AB
        except AttributeError:
            pass
        self._AB = [self.A, self.B]
        return self._AB