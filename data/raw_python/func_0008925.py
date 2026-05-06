def _plot_connectivity(self, A, data=None, lims=[None, None]):
        """
        A debug function used to plot the adjacency/connectivity matrix.
        This is really just a light wrapper around _plot_connectivity_helper
        """

        if data is None:
            data = self.data

        B = A.tocoo()
        self._plot_connectivity_helper(B.col, B.row, B.data, data, lims)