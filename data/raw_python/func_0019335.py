def update(self):
        """Update |WB| based on |RelWB| and |NFk|.

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(2)
        >>> lnk(ACKER)
        >>> relwb(0.2)
        >>> nfk(100.0, 200.0)
        >>> derived.wb.update()
        >>> derived.wb
        wb(20.0, 40.0)
        """
        con = self.subpars.pars.control
        self(con.relwb*con.nfk)