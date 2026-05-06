def get_all_synIdx(self):
        """
        Auxilliary function to set up class attributes containing
        synapse locations given as LFPy.Cell compartment indices

        This function takes no inputs.


        Parameters
        ----------
        None


        Returns
        -------
        synIdx : dict
            `output[cellindex][populationindex][layerindex]` numpy.ndarray of
            compartment indices.


        See also
        --------
        Population.get_synidx, Population.fetchSynIdxCell
        """
        tic = time()

        #containers for synapse idxs existing on this rank
        synIdx = {}


        #ok then, we will draw random numbers across ranks, which have to
        #be unique per cell. Now, we simply record the random state,
        #change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        for cellindex in self.RANK_CELLINDICES:
            #set the random seed on for each cellindex
            np.random.seed(self.POPULATIONSEED + cellindex)

            #find synapse locations for cell in parallel
            synIdx[cellindex] = self.get_synidx(cellindex)

        #reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print('found synapse locations in %.2f seconds' % (time()-tic))

        #print the number of synapses per layer from which presynapse population
        if self.verbose:
            for cellindex in self.RANK_CELLINDICES:
                for i, synidx in enumerate(synIdx[cellindex]):
                    print('to:\t%s\tcell:\t%i\tfrom:\t%s:' % (self.y,
                                                cellindex, self.X[i]),)
                    idxcount = 0
                    for idx in synidx:
                        idxcount += idx.size
                        print('\t%i' % idx.size,)
                    print('\ttotal %i' % idxcount)

        return synIdx