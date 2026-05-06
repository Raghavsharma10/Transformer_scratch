def disaggregate_temperature(self, method='sine_min_max', min_max_time='fix', mod_nighttime=False):
        """
        Disaggregate air temperature.

        Parameters
        ----------
        method : str, optional
            Disaggregation method.

            ``sine_min_max``
                Hourly temperatures follow a sine function preserving daily minimum
                and maximum values. (Default)

            ``sine_mean``
                Hourly temperatures follow a sine function preserving the daily mean
                value and the diurnal temperature range.

            ``sine``
                Same as ``sine_min_max``.

            ``mean_course_min_max``
                Hourly temperatures follow an observed average course (calculated for each month),
                preserving daily minimum and maximum values.

            ``mean_course_mean``
                Hourly temperatures follow an observed average course (calculated for each month),
                preserving the daily mean value and the diurnal temperature range.

        min_max_time : str, optional
            Method to determine the time of minimum and maximum temperature.

            ``fix``:
                Minimum/maximum temperature are assumed to occur at 07:00/14:00 local time.

            ``sun_loc``:
                Minimum/maximum temperature are assumed to occur at sunrise / solar noon + 2 h.

            ``sun_loc_shift``:
                Minimum/maximum temperature are assumed to occur at sunrise / solar noon + monthly mean shift.

        mod_nighttime : bool, optional
            Use linear interpolation between minimum and maximum temperature.
        """
        self.data_disagg.temp = melodist.disaggregate_temperature(
            self.data_daily,
            method=method,
            min_max_time=min_max_time,
            max_delta=self.statistics.temp.max_delta,
            mean_course=self.statistics.temp.mean_course,
            sun_times=self.sun_times,
            mod_nighttime=mod_nighttime
        )