def update(self):
        """Update |KD1| based on |EQD1| and |TInd|.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> eqd1(0.5)
        >>> tind.value = 10.0
        >>> derived.kd1.update()
        >>> derived.kd1
        kd1(5.0)
        """
        con = self.subpars.pars.control
        self(con.eqd1*con.tind)