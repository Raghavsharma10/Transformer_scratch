def update(self):
        """Update |C2| based on :math:`c_2 = 1.-c_1-c_3`.

        Examples:

            The following examples show the calculated value of |C2| are
            clipped when to low or to high:

            >>> from hydpy.models.hstream import *
            >>> parameterstep('1d')
            >>> derived.c1 = 0.6
            >>> derived.c3 = 0.1
            >>> derived.c2.update()
            >>> derived.c2
            c2(0.3)
            >>> derived.c1 = 1.6
            >>> derived.c2.update()
            >>> derived.c2
            c2(0.0)
            >>> derived.c1 = -1.6
            >>> derived.c2.update()
            >>> derived.c2
            c2(1.0)
        """
        der = self.subpars
        self(numpy.clip(1. - der.c1 - der.c3, 0., 1.))