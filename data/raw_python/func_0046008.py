def _gssa(self, initial_conditions, t_max):
        """
        This function is inspired from Yoav Ram's code available at:
        http://nbviewer.ipython.org/github/yoavram/ipython-notebooks/blob/master/GSSA.ipynb

        :param initial_conditions: the initial conditions of the system
        :param t_max:  the time when the simulation should stop
        :return:
        """
        # set the initial conditions and t0 = 0.
        species_over_time = [np.array(initial_conditions).astype("int16")]
        t = 0
        time_points = [t]
        while t < t_max and species_over_time[-1].sum() > 0:
            last = species_over_time[-1]
            e, dt = self._draw(last)
            t += dt
            species_over_time.append(last + self.__change[e,:])
            time_points.append(t)
        return time_points, np.array(species_over_time).T