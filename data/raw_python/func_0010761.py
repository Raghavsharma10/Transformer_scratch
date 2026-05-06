def disaggregate_precipitation(self, method='equal', zerodiv='uniform', shift=0, master_precip=None):
        """
        Disaggregate precipitation.

        Parameters
        ----------
        method : str, optional
            Disaggregation method.

            ``equal``
                Daily precipitation is distributed equally over the 24 hours of the day. (Default)

            ``cascade``
                Hourly precipitation values are obtained using a cascade model set up using
                hourly observations.

        zerodiv : str, optional
            Method to deal with zero division, relevant for ``method='masterstation'``.

            ``uniform``
                Use uniform distribution. (Default)

        master_precip : Series, optional
            Hourly precipitation records from a representative station
            (required for ``method='masterstation'``).
        """
        if method == 'equal':
            precip_disagg = melodist.disagg_prec(self.data_daily, method=method, shift=shift)
        elif method == 'cascade':
            precip_disagg = pd.Series(index=self.data_disagg.index)

            for months, stats in zip(self.statistics.precip.months, self.statistics.precip.stats):
                precip_daily = melodist.seasonal_subset(self.data_daily.precip, months=months)
                if len(precip_daily) > 1:
                    data = melodist.disagg_prec(precip_daily, method=method, cascade_options=stats,
                                                shift=shift, zerodiv=zerodiv)
                    precip_disagg.loc[data.index] = data
        elif method == 'masterstation':
            precip_disagg = melodist.precip_master_station(self.data_daily.precip, master_precip, zerodiv)

        self.data_disagg.precip = precip_disagg