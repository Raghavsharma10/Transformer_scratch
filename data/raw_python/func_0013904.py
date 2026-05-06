def next(self, *arg, **kwarg):
        """Load the next orbit into .data.

        Note
        ----
        Forms complete orbits across day boundaries. If no data loaded
        then the first orbit from the first date of data is returned.
        """

        # first, check if data exists
        if not self.sat.empty:
            # set up orbit metadata
            self._calcOrbits()

            # if current orbit near the last, must be careful
            if self._current == (self.num - 1):
                # first, load last orbit data
                self._getBasicOrbit(orbit=-1)
                # End of orbit may occur on the next day
                load_next = True
                if self.sat._iter_type == 'date':
                    delta = self.sat.date - self.sat.data.index[-1] \
                            + pds.Timedelta('1 day')
                    if delta >= self.orbit_period:
                        # don't need to load the next day because this orbit
                        # ends more than a orbital period from the next date
                        load_next = False

                if load_next:
                    # the end of the user's desired orbit occurs tomorrow, need
                    # to form a complete orbit save this current orbit, load
                    # the next day, combine data, select the correct orbit
                    temp_orbit_data = self.sat.data.copy()
                    try:
                        # loading next day/file clears orbit breaks info
                        self.sat.next()
                        if not self.sat.empty:
                            # combine this next day's data with previous last
                            # orbit, grab the first one
                            self.sat.data = pds.concat(
                                [temp_orbit_data[:self.sat.data.index[0] -
                                                 pds.DateOffset(microseconds=1)],
                                 self.sat.data])
                            self._getBasicOrbit(orbit=1)
                        else:
                            # no data, go back a day and grab the last orbit.
                            # As complete as orbit can be
                            self.sat.prev()
                            self._getBasicOrbit(orbit=-1)
                    except StopIteration:
                        pass
                    del temp_orbit_data
                # includes hack to appear to be zero indexed
                print('Loaded Orbit:%i' % (self._current - 1))

            elif self._current == (self.num):
                # at the last orbit, need to be careful about getting the next
                # orbit save this current orbit and load the next day
                temp_orbit_data = self.sat.data.copy()
                # load next day, which clears orbit breaks info
                self.sat.next()
                # combine this next day orbit with previous last orbit to
                # ensure things are correct
                if not self.sat.empty:
                    pad_next = True
                    # check if data padding is really needed, only works when
                    # loading by date
                    if self.sat._iter_type == 'date':
                        delta = self.sat.date - temp_orbit_data.index[-1]
                        if delta >= self.orbit_period:
                            # the end of the previous orbit is more than an
                            # orbit away from today we don't have to worry
                            # about it
                            pad_next = False
                    if pad_next:
                        # orbit went across day break, stick old orbit onto new
                        # data and grab second orbit (first is old)
                        self.sat.data = pds.concat(
                            [temp_orbit_data[:self.sat.data.index[0] -
                                             pds.DateOffset(microseconds=1)],
                             self.sat.data])
                        # select second orbit of combined data
                        self._getBasicOrbit(orbit=2)
                    else:
                        # padding from the previous orbit wasn't needed, can
                        # just grab the first orbit of loaded data
                        self._getBasicOrbit(orbit=1)
                        if self.sat._iter_type == 'date':
                            delta = self.sat.date + pds.DateOffset(days=1) \
                                    - self.sat.data.index[0]

                            if delta < self.orbit_period:
                                # this orbits end occurs on the next day, though
                                # we grabbed the first orbit, missing data
                                # means the first available orbit in the data
                                # is actually the last for the day. Resetting to
                                # the second to last orbit and then calling
                                # next() will get the last orbit, accounting
                                # for tomorrow's data as well.
                                self._current = self.num - 1
                                self.next()
                else:
                    # no data for the next day
                    # continue loading data until there is some
                    # nextData raises StopIteration when it reaches the end,
                    # leaving this function
                    while self.sat.empty:
                        self.sat.next()
                    self._getBasicOrbit(orbit=1)

                del temp_orbit_data
                # includes hack to appear to be zero indexed
                print('Loaded Orbit:%i' % (self._current - 1))

            elif self._current == 0:
                # no current orbit set, grab the first one
                # using load command to specify the first orbit, which
                # automatically loads prev day if needed to form complete orbit
                self.load(orbit=1)

            elif self._current < (self.num - 1):
                # since we aren't close to the last orbit, just pull the next
                # orbit
                self._getBasicOrbit(orbit=self._current + 1)
                # includes hack to appear to be zero indexed
                print('Loaded Orbit:%i' % (self._current - 1))

            else:
                raise Exception('You ended up where nobody should ever be. ' +
                                'Talk to someone about this fundamental ' +
                                'failure.')

        else:  # no data
            while self.sat.empty:
                # keep going until data is found
                # next raises stopIteration at end of data set, no more data
                # possible
                self.sat.next()
            # we've found data, grab the next orbit
            self.next()