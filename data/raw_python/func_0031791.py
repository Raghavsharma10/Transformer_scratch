def get_all_SpCells(self):
        """
        For each postsynaptic cell existing on this RANK, load or compute
        the presynaptic cell index for each synaptic connection

        This function takes no kwargs.


        Parameters
        ----------
        None


        Returns
        -------
        SpCells : dict
            `output[cellindex][populationname][layerindex]`, np.array of
            presynaptic cell indices.


        See also
        --------
        Population.fetchSpCells

        """
        tic = time()

        #container
        SpCells = {}

        #ok then, we will draw random numbers across ranks, which have to
        #be unique per cell. Now, we simply record the random state,
        #change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        for cellindex in self.RANK_CELLINDICES:
            #set the random seed on for each cellindex
            np.random.seed(self.POPULATIONSEED + cellindex + self.POPULATION_SIZE)

            SpCells[cellindex] = {}
            for i, X in enumerate(self.X):
                SpCells[cellindex][X] = self.fetchSpCells(
                    self.networkSim.nodes[X], self.k_yXL[:, i])

        #reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print('found presynaptic cells in %.2f seconds' % (time()-tic))

        return SpCells