def _orbitNumberBreaks(self):
        """Determine where orbital breaks in a dataset with orbit numbers occur.

        Looks for changes in unique values.

        """

        if self.orbit_index is None:
            raise ValueError('Orbit properties must be defined at ' +
                             'pysat.Instrument object instantiation.' +
                             'See Instrument docs.')
        else:
            try:
                self.sat[self.orbit_index]
            except ValueError:
                raise ValueError('Provided orbit index does not appear to ' +
                                 'exist in loaded data')

        # determine where the orbit index changes from one value to the next
        uniq_vals = self.sat[self.orbit_index].unique()
        orbit_index = []
        for val in uniq_vals:
            idx, = np.where(val == self.sat[self.orbit_index].values)
            orbit_index.append(idx[0])

        # create orbitbreak index, ensure first element is always 0
        if orbit_index[0] != 0:
            ind = np.hstack((np.array([0]), orbit_index))
        else:
            ind = orbit_index
        # number of orbits
        num_orbits = len(ind)
        # set index of orbit breaks
        self._orbit_breaks = ind
        # set number of orbits for the day
        self.num = num_orbits