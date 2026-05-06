def __diff_internal(self):
        """Differentiate a B-spline once, and return the resulting coefficients and Bspline objects.

This preserves the Bspline object nature of the data, enabling recursive implementation
of higher-order differentiation (see `diff`).

The value of the first derivative of `B` at a point `x` can be obtained as::

    def diff1(B, x):
        terms = B.__diff_internal()
        return sum( ci*Bi(x) for ci,Bi in terms )

Returns:
    tuple of tuples, where each item is (coefficient, Bspline object).

See:
    `diff`: differentiation of any order >= 0
"""
        assert self.p > 0, "order of Bspline must be > 0"  # we already handle the other case in diff()

        # https://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
        #
        t    = self.knot_vector
        p    = self.p
        Bi   = Bspline( t[:-1], p-1 )
        Bip1 = Bspline( t[1:],  p-1 )

        numer1 = +p
        numer2 = -p
        denom1 = t[p:-1]   - t[:-(p+1)]
        denom2 = t[(p+1):] - t[1:-p]

        with np.errstate(divide='ignore', invalid='ignore'):
            ci   = np.where(denom1 != 0., (numer1 / denom1), 0.)
            cip1 = np.where(denom2 != 0., (numer2 / denom2), 0.)

        return ( (ci,Bi), (cip1,Bip1) )