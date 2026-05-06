def isCollinear(self, b, c):
        '''
        :b: Point or point equivalent
        :c: Point or point equivalent
        :return: boolean

        True if 'self' is collinear with 'b' and 'c', otherwise False.
        '''

        return all(self.ccw(b, c, axis) == 0 for axis in self._keys)