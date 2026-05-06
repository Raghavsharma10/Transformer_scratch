def _calcOrbits(self):
        """Prepares data structure for breaking data into orbits. Not intended
        for end user."""
        # if the breaks between orbit have not been defined, define them
        # also, store the data so that grabbing different orbits does not
        # require reloads of whole dataset
        if len(self._orbit_breaks) == 0:
            # determine orbit breaks
            self._detBreaks()
            # store a copy of data
            self._fullDayData = self.sat.data.copy()
            # set current orbit counter to zero (default)
            self._current = 0