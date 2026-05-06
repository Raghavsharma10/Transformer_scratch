def _tarboton_slopes_directions(self, data, dX, dY):
        """
        Calculate the slopes and directions based on the 8 sections from
        Tarboton http://www.neng.usu.edu/cee/faculty/dtarb/96wr03137.pdf
        """

        return _tarboton_slopes_directions(data, dX, dY,
                                           self.facets, self.ang_adj)