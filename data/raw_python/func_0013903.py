def load(self, orbit=None):
        """Load a particular orbit into .data for loaded day.

        Parameters
        ----------
        orbit : int
            orbit number, 1 indexed

        Note
        ----    
        A day of data must be loaded before this routine functions properly.
        If the last orbit of the day is requested, it will automatically be
        padded with data from the next day. The orbit counter will be 
        reset to 1.
        """
        if not self.sat.empty:  # ensure data exists
            # set up orbit metadata
            self._calcOrbits()
            # ensure user supplied an orbit
            if orbit is not None:
                # pull out requested orbit
                if orbit < 0:
                    # negative indexing consistent with numpy, -1 last,
                    # -2 second to last, etc.
                    orbit = self.num + 1 + orbit

                if orbit == 1:
                    # change from orig copied from _core, didn't look correct.
                    # self._getBasicOrbit(orbit=2)
                    try:
                        true_date = self.sat.date  # .copy()

                        self.sat.prev()
                        # if and else added becuase of CINDI turn off 
                        # 6/5/2013, turn on 10/22/2014
                        # crashed when starting on 10/22/2014
                        # prev returned empty data
                        if not self.sat.empty:
                            self.load(orbit=-1)
                        else:
                            self.sat.next()
                            self._getBasicOrbit(orbit=1)
                        # check that this orbit should end on the current day
                        delta = true_date - self.sat.data.index[0]
                        # print 'checking if first orbit should land on requested day'
                        # print self.sat.date, self.sat.data.index[0], delta, delta >= self.orbit_period
                        if delta >= self.orbit_period:
                            # the orbit loaded isn't close enough to date
                            # to be the first orbit of the day, move forward
                            self.next()
                    except StopIteration:
                        # print 'going for basic orbit'
                        self._getBasicOrbit(orbit=1)
                        # includes hack to appear to be zero indexed
                        print('Loaded Orbit:%i' % (self._current - 1))
                        # check if the first orbit is also the last orbit

                elif orbit == self.num:
                    # we get here if user asks for last orbit
                    # make sure that orbit data goes across daybreak as needed
                    # load previous orbit
                    if self.num != 1:
                        self._getBasicOrbit(self.num - 1)
                        self.next()
                    else:
                        self._getBasicOrbit(orbit=-1)

                elif orbit < self.num:
                    # load orbit data into data
                    self._getBasicOrbit(orbit)
                    # includes hack to appear to be zero indexed
                    print('Loaded Orbit:%i' % (self._current - 1))

                else:
                    # gone too far
                    self.sat.data = DataFrame()
                    raise Exception('Requested an orbit past total orbits for day')
            else:
                raise Exception('Must set an orbit')
        else:
            print('No data loaded in instrument object to determine orbits.')