def update(self):
        """Update |TTM| based on :math:`TTM = TT+DTTM`.

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(1)
        >>> zonetype(FIELD)
        >>> tt(1.0)
        >>> dttm(-2.0)
        >>> derived.ttm.update()
        >>> derived.ttm
        ttm(-1.0)
        """
        con = self.subpars.pars.control
        self(con.tt+con.dttm)