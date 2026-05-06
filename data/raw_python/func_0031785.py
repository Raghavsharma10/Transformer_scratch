def set_pop_soma_pos(self):
        """
        Set `pop_soma_pos` using draw_rand_pos().

        This method takes no keyword arguments.


        Parameters
        ----------
        None


        Returns
        -------
        numpy.ndarray
            (x,y,z) coordinates of each neuron in the population


        See also
        --------
        PopulationSuper.draw_rand_pos

        """
        tic = time()
        if RANK == 0:
            pop_soma_pos = self.draw_rand_pos(
                min_r = self.electrodeParams['r_z'],
                **self.populationParams)
        else:
            pop_soma_pos = None

        if RANK == 0:
            print('found cell positions in %.2f s' % (time()-tic))

        return COMM.bcast(pop_soma_pos, root=0)