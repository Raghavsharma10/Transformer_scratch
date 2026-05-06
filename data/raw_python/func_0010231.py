def ABC(self):
        '''
        A list of the triangle's vertices, list.

        '''
        try:
            return self._ABC
        except AttributeError:
            pass
        self._ABC = [self.A, self.B, self.C]
        return self._ABC