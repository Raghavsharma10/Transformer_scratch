def _polarBreaks(self):
        """Determine where breaks in a polar orbiting satellite orbit occur.

        Looks for sign changes in latitude (magnetic or geographic) as well as 
        breaks in UT.
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

        # determine where orbit index goes from positive to negative
        pos = (self.sat[self.orbit_index] >= 0)
        npos = -pos
        change = (pos.values[:-1] & npos.values[1:]) | (npos.values[:-1] &
                                                        pos.values[1:])

        ind, = np.where(change)
        ind += 1

        ut_diff = Series(self.sat.data.index).diff()
        ut_ind, = np.where(ut_diff / self.orbit_period > 0.95)

        if len(ut_ind) > 0:
            ind = np.hstack((ind, ut_ind))
            ind = np.sort(ind)
            ind = np.unique(ind)
            # print 'Time Gap'

        # create orbitbreak index, ensure first element is always 0
        if ind[0] != 0:
            ind = np.hstack((np.array([0]), ind))
        # number of orbits
        num_orbits = len(ind)
        # set index of orbit breaks
        self._orbit_breaks = ind
        # set number of orbits for the day
        self.num = num_orbits