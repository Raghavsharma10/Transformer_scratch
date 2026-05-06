def _equaBreaks(self, orbit_index_period=24.):
        """Determine where breaks in an equatorial satellite orbit occur.

        Looks for negative gradients in local time (or longitude) as well as 
        breaks in UT.

        Parameters
        ----------
        orbit_index_period : float
           The change in value of supplied index parameter for a single orbit
        """

        if self.orbit_index is None:
            raise ValueError('Orbit properties must be defined at ' +
                             'pysat.Instrument object instantiation.' + 
                             'See Instrument docs.')
        else:
            try:
                self.sat[self.orbit_index]
            except ValueError:
                raise ValueError('Provided orbit index does not exist in ' +
                                 'loaded data')
        # get difference in orbit index around the orbit
        lt_diff = self.sat[self.orbit_index].diff()
        # universal time values, from datetime index
        ut_vals = Series(self.sat.data.index)
        # UT difference
        ut_diff = ut_vals.diff()

        # get locations where orbit index derivative is less than 0
        # then do some basic checks on these locations
        ind, = np.where((lt_diff < -0.1))
        if len(ind) > 0:
            ind = np.hstack((ind, np.array([len(self.sat[self.orbit_index])])))
            # look at distance between breaks
            dist = ind[1:] - ind[0:-1]
            # only keep orbit breaks with a distance greater than 1
            # done for robustness
            if len(ind) > 1:
                if min(dist) == 1:
                    print('There are orbit breaks right next to each other')
                ind = ind[:-1][dist > 1]

            # check for large positive gradients around the break that would
            # suggest not a true orbit break, but rather bad orbit_index values
            new_ind = []
            for idx in ind:
                tidx, = np.where(lt_diff[idx - 5:idx + 6] > 0.1)

                if len(tidx) != 0:
                    # there are large changes, suggests a false alarm
                    # iterate over samples and check
                    for tidx in tidx:
                        # look at time change vs local time change
                        if(ut_diff[idx - 5:idx + 6].iloc[tidx] <
                           lt_diff[idx - 5:idx + 6].iloc[tidx] /
                           orbit_index_period * self.orbit_period):
                            # change in ut is small compared to the change in
                            # the orbit index this is flagged as a false alarm,
                            # or dropped from consideration
                            pass
                        else:
                            # change in UT is significant, keep orbit break
                            new_ind.append(idx)
                            break
                else:
                    # no large positive gradients, current orbit break passes
                    # the first test
                    new_ind.append(idx)
            # replace all breaks with those that are 'good'
            ind = np.array(new_ind)

        # now, assemble some orbit breaks that are not triggered by changes in
        # the orbit index

        # check if there is a UT break that is larger than orbital period, aka
        # a time gap
        ut_change_vs_period = ( ut_diff > self.orbit_period )
        # characterize ut change using orbital period
        norm_ut = ut_diff / self.orbit_period
        # now, look for breaks because the length of time between samples is
        # too large, thus there is no break in slt/mlt/etc, lt_diff is small
        # but UT change is big
        norm_ut_vs_norm_lt = norm_ut.gt(np.abs(lt_diff.values /
                                               orbit_index_period))
        # indices when one or other flag is true
        ut_ind, = np.where(ut_change_vs_period | (norm_ut_vs_norm_lt &
                                                  (norm_ut > 0.95)))
        # added the or and check after or on 10/20/2014
        # & lt_diff.notnull() ))# & (lt_diff != 0)  ) )

        # combine these UT determined orbit breaks with the orbit index orbit
        # breaks
        if len(ut_ind) > 0:
            ind = np.hstack((ind, ut_ind))
            ind = np.sort(ind)
            ind = np.unique(ind)
            print('Time Gap')

        # now that most problems in orbits should have been caught, look at
        # the time difference between orbits (not individual orbits)
        orbit_ut_diff = ut_vals[ind].diff()
        orbit_lt_diff = self.sat[self.orbit_index][ind].diff()
        # look for time gaps between partial orbits. The full orbital time
        # period is not required between end of one orbit and begining of next
        # if first orbit is partial.  Also provides another general test of the
        # orbital breaks determined.
        idx, = np.where((orbit_ut_diff / self.orbit_period -
                         orbit_lt_diff.values / orbit_index_period) > 0.97)
        # pull out breaks that pass the test, need to make sure the first one
        # is always included it gets dropped via the nature of diff
        if len(idx) > 0:
            if idx[0] != 0:
                idx = np.hstack((0, idx))
        else:
            idx = np.array([0])
        # only keep the good indices
        if len(ind) > 0:
            ind = ind[idx]
            # create orbitbreak index, ensure first element is always 0
            if ind[0] != 0:
                ind = np.hstack((np.array([0]), ind))
        else:
            ind = np.array([0])
        # number of orbits
        num_orbits = len(ind)
        # set index of orbit breaks
        self._orbit_breaks = ind
        # set number of orbits for the day
        self.num = num_orbits