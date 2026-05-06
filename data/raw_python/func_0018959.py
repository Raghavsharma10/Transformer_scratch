def update(self):
        """Update value based on :math:`HV=BBV/BNV`.

        Required Parameters:
            |BBV|
            |BNV|

        Examples:
            >>> from hydpy.models.lstream import *
            >>> parameterstep('1d')
            >>> bbv(left=10., right=40.)
            >>> bnv(left=10., right=20.)
            >>> derived.hv.update()
            >>> derived.hv
            hv(left=1.0, right=2.0)
            >>> bbv(left=10., right=0.)
            >>> bnv(left=0., right=20.)
            >>> derived.hv.update()
            >>> derived.hv
            hv(0.0)
        """
        con = self.subpars.pars.control
        self(0.)
        for idx in range(2):
            if (con.bbv[idx] > 0.) and (con.bnv[idx] > 0.):
                self.values[idx] = con.bbv[idx]/con.bnv[idx]