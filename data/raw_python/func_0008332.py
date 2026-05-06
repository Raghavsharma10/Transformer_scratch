def update(self):
        """
        Updates @ 10Hz to give smooth running clock.
        """

        try:

            # update counter
            self.counter += 1
            g = get_root(self).globals

            # current time
            now = Time.now()

            # configure times
            self.utc.configure(text=now.datetime.strftime('%H:%M:%S'))
            self.mjd.configure(text='{0:11.5f}'.format(now.mjd))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # ignore astropy deprecation warnings
                lon = self.obs.longitude
            lst = now.sidereal_time(kind='mean', longitude=lon)
            self.lst.configure(text=lst.to_string(sep=':', precision=0))

            if self.counter % 600 == 1:
                # only re-compute Sun & Moon info once every 600 calls
                altaz_frame = coord.AltAz(obstime=now, location=self.obs)
                sun = coord.get_sun(now)
                sun_aa = sun.transform_to(altaz_frame)
                moon = coord.get_moon(now, self.obs)
                moon_aa = moon.transform_to(altaz_frame)
                elongation = sun.separation(moon)
                moon_phase_angle = np.arctan2(sun.distance*np.sin(elongation),
                                              moon.distance - sun.distance*np.cos(elongation))
                moon_phase = (1 + np.cos(moon_phase_angle))/2.0

                self.sunalt.configure(
                    text='{0:+03d} deg'.format(int(sun_aa.alt.deg))
                )
                self.moonra.configure(
                    text=moon.ra.to_string(unit='hour', sep=':', precision=0)
                )
                self.moondec.configure(
                    text=moon.dec.to_string(unit='deg', sep=':', precision=0)
                )
                self.moonalt.configure(text='{0:+03d} deg'.format(
                        int(moon_aa.alt.deg)
                ))
                self.moonphase.configure(text='{0:02d} %'.format(
                        int(100.*moon_phase.value)
                ))

                if (now > self.lastRiset and now > self.lastAstro):
                    # Only re-compute rise and setting times when necessary,
                    # and only re-compute when both rise/set and astro
                    # twilight times have gone by

                    # For sunrise and set we set the horizon down to match a
                    # standard amount of refraction at the horizon and subtract size of disc
                    horizon = -64*u.arcmin
                    sunset = calc_riseset(now, 'sun', self.obs, 'next', 'setting', horizon)
                    sunrise = calc_riseset(now, 'sun', self.obs, 'next', 'rising', horizon)

                    # Astro twilight: geometric centre at -18 deg
                    horizon = -18*u.deg
                    astroset = calc_riseset(now, 'sun', self.obs, 'next', 'setting', horizon)
                    astrorise = calc_riseset(now, 'sun', self.obs, 'next', 'rising', horizon)

                    if sunrise > sunset:
                        # In the day time we report the upcoming sunset and
                        # end of evening twilight
                        self.lriset.configure(text='Sets:', font=g.DEFAULT_FONT)
                        self.lastRiset = sunset
                        self.lastAstro = astroset

                    elif astrorise > astroset and astrorise < sunrise:
                        # During evening twilight, we report the sunset just
                        # passed and end of evening twilight
                        self.lriset.configure(text='Sets:', font=g.DEFAULT_FONT)
                        horizon = -64*u.arcmin
                        self.lastRiset = calc_riseset(now, 'sun', self.obs, 'previous', 'setting', horizon)
                        self.lastAstro = astroset

                    elif astrorise > astroset and astrorise < sunrise:
                        # During night, report upcoming start of morning
                        # twilight and sunrise
                        self.lriset.configure(text='Rises:',
                                              font=g.DEFAULT_FONT)
                        horizon = -64*u.arcmin
                        self.lastRiset = sunrise
                        self.lastAstro = astrorise

                    else:
                        # During morning twilight report start of twilight
                        # just passed and upcoming sunrise
                        self.lriset.configure(text='Rises:',
                                              font=g.DEFAULT_FONT)
                        horizon = -18*u.deg
                        self.lastRiset = sunrise
                        self.lastAstro = calc_riseset(now, 'sun', self.obs, 'previous', 'rising', horizon)

                    # Configure the corresponding text fields
                    self.riset.configure(
                        text=self.lastRiset.datetime.strftime("%H:%M:%S")
                    )
                    self.astro.configure(
                        text=self.lastAstro.datetime.strftime("%H:%M:%S")
                    )

        except Exception as err:
            # catchall
            g.clog.warn('AstroFrame.update: error = ' + str(err))

        # run again after 100 milli-seconds
        self.after(100, self.update)