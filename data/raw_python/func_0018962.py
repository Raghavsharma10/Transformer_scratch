def update(self):
        """Update |C1| based on :math:`c_1 = \\frac{Damp}{1+Damp}`.

        Examples:

            The first examples show the calculated value of |C1| for
            the lowest possible value of |Lag|, the lowest possible value,
            and an intermediate value:

            >>> from hydpy.models.hstream import *
            >>> parameterstep('1d')
            >>> damp(0.0)
            >>> derived.c1.update()
            >>> derived.c1
            c1(0.0)
            >>> damp(1.0)
            >>> derived.c1.update()
            >>> derived.c1
            c1(0.5)
            >>> damp(0.25)
            >>> derived.c1.update()
            >>> derived.c1
            c1(0.2)

            For to low and to high values of |Lag|, clipping is performed:
            >>> damp.value = -0.1
            >>> derived.c1.update()
            >>> derived.c1
            c1(0.0)
            >>> damp.value = 1.1
            >>> derived.c1.update()
            >>> derived.c1
            c1(0.5)
        """
        damp = self.subpars.pars.control.damp
        self(numpy.clip(damp/(1.+damp), 0., .5))