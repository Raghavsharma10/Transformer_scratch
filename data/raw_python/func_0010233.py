def AC(self):
        '''
        Vertices A and C, list.

        '''
        try:
            return self._AC
        except AttributeError:
            pass
        self._AC = [self.A, self.C]
        return self._AC