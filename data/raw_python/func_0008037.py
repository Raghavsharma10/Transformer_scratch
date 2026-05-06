def diff(self, order=1):
        """Differentiate a B-spline `order` number of times.

Parameters:
    order:
        int, >= 0

Returns:
    **lambda** `x`: ... that evaluates the `order`-th derivative of `B` at the point `x`.
                    The returned function internally uses __call__, which is 'memoized' for speed.
"""
        order = int(order)
        if order < 0:
            raise ValueError("order must be >= 0, got %d" % (order))

        if order == 0:
            return self.__call__

        if order > self.p:   # identically zero, but force the same output format as in the general case
            dummy = self.__call__(0.)  # get number of basis functions and output dtype
            nbasis = dummy.shape[0]
            return lambda x: np.zeros( (nbasis,), dtype=dummy.dtype )  # accept but ignore input x

        # At each differentiation, each term maps into two new terms.
        # The number of terms in the result will be 2**order.
        #
        # This will cause an exponential explosion in the number of terms for high derivative orders,
        # but for the first few orders (practical usage; >3 is rarely needed) the approach works.
        #
        terms = [ (1.,self) ]
        for k in range(order):
            tmp = []
            for Ci,Bi in terms:
                tmp.extend( (Ci*cn, Bn) for cn,Bn in Bi.__diff_internal() )  # NOTE: also propagate Ci
            terms = tmp

        # perform final summation at call time
        return lambda x: sum( ci*Bi(x) for ci,Bi in terms )