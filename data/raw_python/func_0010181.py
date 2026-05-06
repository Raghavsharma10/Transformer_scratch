def _unary_(self, func, inplace=False):
        '''
        :func: unary function to apply to each coordinate
        :inplace: optional boolean
        :return: Point

        Implementation private method.

        All of the unary operations funnel thru this method
        to reduce cut-and-paste code and enforce consistent
        behavior of unary ops.

        Applies 'func' to self and returns the result.

        The expected call signature of 'func' is f(a)

        If 'inplace' is True, the results are stored in 'self',
        otherwise the results will be stored in a new object.

        Returns a Point.

        '''
        dst = self if inplace else self.__class__(self)
        dst.x = func(dst.x)
        dst.y = func(dst.y)
        dst.z = func(dst.z)
        return dst