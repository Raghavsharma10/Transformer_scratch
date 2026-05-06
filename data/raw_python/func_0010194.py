def center(self):
        '''
        Center point of the ellipse, equidistant from foci, Point class.\n
        Defaults to the origin.
        '''
        try:
            return self._center
        except AttributeError:
            pass
        self._center = Point()
        return self._center