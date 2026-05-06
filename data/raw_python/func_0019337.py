def update(self):
        """Update |KB| based on |EQB| and |TInd|.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> eqb(10.0)
        >>> tind.value = 10.0
        >>> derived.kb.update()
        >>> derived.kb
        kb(100.0)
        """
        con = self.subpars.pars.control
        self(con.eqb*con.tind)