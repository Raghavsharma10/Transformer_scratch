def _binary_(self, other, func, inplace=False):
        '''
        :other:   Point or point equivalent
        :func:    binary function to apply
        :inplace: optional boolean
        :return:  Point

        Implementation private method.

        All of the binary operations funnel thru this method to
        reduce cut-and-paste code and enforce consistent behavior
        of binary ops.

        Applies 'func' to 'self' and 'other' and returns the result.

        If 'inplace' is True the results of will be stored in 'self',
        otherwise the results will be stored in a new object.

        Returns a Point.

        '''

        dst = self if inplace else self.__class__(self)

        try:
            b = self.__class__._convert(other, ignoreScalars=True)
            dst.x = func(dst.x, b.x)
            dst.y = func(dst.y, b.y)
            dst.z = func(dst.z, b.z)
            return dst
        except TypeError:
            pass

        dst.x = func(dst.x, other)
        dst.y = func(dst.y, other)
        dst.z = func(dst.z, other)
        return dst