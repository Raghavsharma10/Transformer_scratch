def _check_date(self, fmt='%Y-%m-%d'):
        """Convenience method for determining if the input date is valid.

        Parameters
        ----------
        fmt : str
            The format of the date string. The default is ``%Y-%m-%d``, which
            corresponds to ``YYYY-MM-DD``.

        Returns
        -------
        status : str or `None`
            If the date is valid, returns `None`. If the date is invalid,
            returns a message explaining the issue.

        """
        result = None
        try:
            dt_obj = dt.datetime.strptime(self.date, fmt)
        except ValueError:
            result = '{} does not match YYYY-MM-DD format'.format(self.date)
        else:
            if dt_obj < self._acs_installation_date:
                result = ('The observation date cannot occur '
                          'before ACS was installed ({})'
                          .format(self._acs_installation_date.strftime(fmt)))
            elif dt_obj > self._extrapolation_date:
                result = ('The observation date cannot occur after the '
                          'maximum allowable date, {}. Extrapolations of the '
                          'instrument throughput after this date lead to '
                          'high uncertainties and are therefore invalid.'
                          .format(self._extrapolation_date.strftime(fmt)))
        finally:
            return result