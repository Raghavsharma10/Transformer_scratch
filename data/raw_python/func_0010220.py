def origin(self):
        '''
        Point describing the origin of the rectangle. Defaults to (0,0,0).
        '''
        try:
            return self._origin
        except AttributeError:
            pass
        self._origin = Point()
        return self._origin