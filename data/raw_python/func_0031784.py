def run(self):
        """
        Distribute individual cell simulations across ranks.

        This method takes no keyword arguments.


        Parameters
        ----------
        None


        Returns
        -------
        None

        """
        for cellindex in self.RANK_CELLINDICES:
            self.cellsim(cellindex)

        COMM.Barrier()