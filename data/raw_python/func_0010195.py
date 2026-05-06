def radius(self):
        '''
        Radius of the ellipse, Point class.
        '''
        try:
            return self._radius
        except AttributeError:
            pass
        self._radius = Point(1, 1, 0)
        return self._radius