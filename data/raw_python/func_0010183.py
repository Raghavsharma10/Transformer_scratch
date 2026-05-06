def isBetween(self, a, b, axes='xyz'):
        '''
        :a: Point or point equivalent
        :b: Point or point equivalent
        :axis: optional string
        :return: float

        Checks the coordinates specified in 'axes' of 'self' to
        determine if they are bounded by 'a' and 'b'. The range
        is inclusive of end-points.

        Returns boolean.
        '''
        a = self.__class__._convert(a)
        b = self.__class__._convert(b)

        fn = lambda k: (self[k] >= min(a[k], b[k])) and (
            self[k] <= max(a[k], b[k]))

        return all(fn(axis) for axis in axes)