def update(self):
        """Determine the number of branches"""
        con = self.subpars.pars.control
        self(con.ypoints.shape[0])