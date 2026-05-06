def draw_rand_pos(self, radius, z_min, z_max,
                      min_r=np.array([0]), min_cell_interdist=10., **args):
        """
        Draw some random location within radius, z_min, z_max,
        and constrained by min_r and the minimum cell interdistance.
        Returned argument is a list of dicts [{'xpos', 'ypos', 'zpos'},].


        Parameters
        ----------
        radius : float
            Radius of population.
        z_min : float
            Lower z-boundary of population.
        z_max : float
            Upper z-boundary of population.
        min_r : numpy.ndarray
            Minimum distance to center axis as function of z.
        min_cell_interdist : float
            Minimum cell to cell interdistance.
        **args : keyword arguments
            Additional inputs that is being ignored.


        Returns
        -------
        soma_pos : list
            List of dicts of len population size
            where dict have keys xpos, ypos, zpos specifying
            xyz-coordinates of cell at list entry `i`.


        See also
        --------
        PopulationSuper.calc_min_cell_interdist

        """
        x = (np.random.rand(self.POPULATION_SIZE)-0.5)*radius*2
        y = (np.random.rand(self.POPULATION_SIZE)-0.5)*radius*2
        z = np.random.rand(self.POPULATION_SIZE)*(z_max - z_min) + z_min
        min_r_z = {}
        min_r = np.array(min_r)
        if min_r.size > 0:
            if type(min_r) == type(np.array([])):
                j = 0
                for j in range(min_r.shape[0]):
                    min_r_z[j] = np.interp(z, min_r[0,], min_r[1,])
                    if j > 0:
                        [w] = np.where(min_r_z[j] < min_r_z[j-1])
                        min_r_z[j][w] = min_r_z[j-1][w]
                minrz = min_r_z[j]
        else:
            minrz = np.interp(z, min_r[0], min_r[1])

        R_z = np.sqrt(x**2 + y**2)

        #want to make sure that no somas are in the same place.
        cell_interdist = self.calc_min_cell_interdist(x, y, z)

        [u] = np.where(np.logical_or((R_z < minrz) != (R_z > radius),
            cell_interdist < min_cell_interdist))

        while len(u) > 0:
            for i in range(len(u)):
                x[u[i]] = (np.random.rand()-0.5)*radius*2
                y[u[i]] = (np.random.rand()-0.5)*radius*2
                z[u[i]] = np.random.rand()*(z_max - z_min) + z_min
                if type(min_r) == type(()):
                    for j in range(np.shape(min_r)[0]):
                        min_r_z[j][u[i]] = \
                                np.interp(z[u[i]], min_r[0,], min_r[1,])
                        if j > 0:
                            [w] = np.where(min_r_z[j] < min_r_z[j-1])
                            min_r_z[j][w] = min_r_z[j-1][w]
                        minrz = min_r_z[j]
                else:
                    minrz[u[i]] = np.interp(z[u[i]], min_r[0,], min_r[1,])
            R_z = np.sqrt(x**2 + y**2)

            #want to make sure that no somas are in the same place.
            cell_interdist = self.calc_min_cell_interdist(x, y, z)

            [u] = np.where(np.logical_or((R_z < minrz) != (R_z > radius),
                cell_interdist < min_cell_interdist))

        
        soma_pos = []
        for i in range(self.POPULATION_SIZE):
            soma_pos.append({'xpos' : x[i], 'ypos' : y[i], 'zpos' : z[i]})

        return soma_pos