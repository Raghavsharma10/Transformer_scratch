def C(self):
        '''
        Third vertex of triangle, Point subclass.

        '''
        try:
            return self._C
        except AttributeError:
            pass
        self._C = Point(0, 1)
        return self._C