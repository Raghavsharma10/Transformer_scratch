def disaggregate_radiation(self, method='pot_rad', pot_rad=None):
        """
        Disaggregate solar radiation.

        Parameters
        ----------
        method : str, optional
            Disaggregation method.

            ``pot_rad``
                Calculates potential clear-sky hourly radiation and scales it according to the
                mean daily radiation. (Default)

            ``pot_rad_via_ssd``
                Calculates potential clear-sky hourly radiation and scales it according to the
                observed daily sunshine duration.

            ``pot_rad_via_bc``
                Calculates potential clear-sky hourly radiation and scales it according to daily
                minimum and maximum temperature.

            ``mean_course``
                Hourly radiation follows an observed average course (calculated for each month).

        pot_rad : Series, optional
            Hourly values of potential solar radiation. If ``None``, calculated internally.
        """
        if self.sun_times is None:
            self.calc_sun_times()

        if pot_rad is None and method != 'mean_course':
            pot_rad = melodist.potential_radiation(self.data_disagg.index, self.lon, self.lat, self.timezone)

        self.data_disagg.glob = melodist.disaggregate_radiation(
            self.data_daily,
            sun_times=self.sun_times,
            pot_rad=pot_rad,
            method=method,
            angstr_a=self.statistics.glob.angstroem.a,
            angstr_b=self.statistics.glob.angstroem.b,
            bristcamp_a=self.statistics.glob.bristcamp.a,
            bristcamp_c=self.statistics.glob.bristcamp.c,
            mean_course=self.statistics.glob.mean_course
        )