def get_all_synDelays(self):
        """
        Create and load arrays of connection delays per connection on this rank

        Get random normally distributed synaptic delays,
        returns dict of nested list of same shape as SpCells.

        Delays are rounded to dt.

        This function takes no kwargs.


        Parameters
        ----------
        None


        Returns
        -------
        dict
            output[cellindex][populationname][layerindex]`, np.array of
            delays per connection.


        See also
        --------
        numpy.random.normal

        """
        tic = time()

        #ok then, we will draw random numbers across ranks, which have to
        #be unique per cell. Now, we simply record the random state,
        #change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        #container
        delays = {}

        for cellindex in self.RANK_CELLINDICES:
            #set the random seed on for each cellindex
            np.random.seed(self.POPULATIONSEED + cellindex + 2*self.POPULATION_SIZE)

            delays[cellindex] = {}
            for j, X in enumerate(self.X):
                delays[cellindex][X] = []
                for i in self.k_yXL[:, j]:
                    loc = self.synDelayLoc[j]
                    loc /= self.dt
                    scale = self.synDelayScale[j]
                    if scale is not None:
                        scale /= self.dt
                        delay = np.random.normal(loc, scale, i).astype(int)
                        while np.any(delay < 1):
                            inds = delay < 1
                            delay[inds] = np.random.normal(loc, scale,
                                                        inds.sum()).astype(int)
                        delay = delay.astype(float)
                        delay *= self.dt
                    else:
                        delay = np.zeros(i) + self.synDelayLoc[j]
                    delays[cellindex][X].append(delay)

        #reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print('found delays in %.2f seconds' % (time()-tic))

        return delays