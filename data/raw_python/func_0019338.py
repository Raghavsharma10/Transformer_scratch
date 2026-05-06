def update(self):
        """Update |KI1| based on |EQI1| and |TInd|.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> eqi1(5.0)
        >>> tind.value = 10.0
        >>> derived.ki1.update()
        >>> derived.ki1
        ki1(50.0)
        """
        con = self.subpars.pars.control
        self(con.eqi1*con.tind)