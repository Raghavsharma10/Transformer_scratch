def _getBasicOrbit(self, orbit=None):
        """Load a particular orbit into .data for loaded day.

        Parameters
        ----------
        orbit : int
            orbit number, 1 indexed, negative indexes allowed, -1 last orbit

        Note
        ----
        A day of data must be loaded before this routine functions properly.
        If the last orbit of the day is requested, it will NOT automatically be
        padded with data from the next day.
        """
        # ensure data exists
        if not self.sat.empty:
            # ensure proper orbit metadata present
            self._calcOrbits()

            # ensure user is requesting a particular orbit
            if orbit is not None:
                # pull out requested orbit
                if orbit == -1:
                    # load orbit data into data
                    self.sat.data = self._fullDayData[self._orbit_breaks[self.num + orbit]:]
                    self._current = self.num + orbit + 1
                elif ((orbit < 0) & (orbit >= -self.num)):
                    # load orbit data into data
                    self.sat.data = self._fullDayData[
                                    self._orbit_breaks[self.num + orbit]:self._orbit_breaks[self.num + orbit + 1]]
                    self._current = self.num + orbit + 1
                elif (orbit < self.num) & (orbit != 0):
                    # load orbit data into data
                    self.sat.data = self._fullDayData[self._orbit_breaks[orbit - 1]:self._orbit_breaks[orbit]]
                    self._current = orbit
                elif orbit == self.num:
                    self.sat.data = self._fullDayData[self._orbit_breaks[orbit - 1]:]
                    # recent addition, wondering why it wasn't there before,
                    # could just be a bug that is now fixed.
                    self._current = orbit
                elif orbit == 0:
                    raise ValueError('Orbits internally indexed by 1, 0 not ' +
                                     'allowed')
                else:
                    # gone too far
                    self.sat.data = []
                    raise ValueError('Requested an orbit past total orbits ' +
                                     'for day')
            else:
                raise ValueError('Must set an orbit')