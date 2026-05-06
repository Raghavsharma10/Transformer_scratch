def ccw(self, b, c, axis='z'):
        '''
        :b: Point or point equivalent
        :c: Point or point equivalent
        :axis: optional string or integer in set('x',0,'y',1,'z',2)
        :return: float

        CCW - Counter Clockwise

        Returns an integer signifying the direction of rotation around 'axis'
        described by the angle [b, self, c].

        > 0 : counter-clockwise
          0 : points are collinear
        < 0 : clockwise

        Returns an integer.

        Raises ValueError if axis is not in 'xyz'.
        '''
        bsuba = b - self
        csuba = c - self

        if axis in ['z', 2]:
            return (bsuba.x * csuba.y) - (bsuba.y * csuba.x)

        if axis in ['y', 1]:
            return (bsuba.x * csuba.z) - (bsuba.z * csuba.x)

        if axis in ['x', 0]:
            return (bsuba.y * csuba.z) - (bsuba.z * csuba.y)

        msg = "invalid axis '{!r}', must be one of {}".format(axis, self._keys)

        raise ValueError(msg)