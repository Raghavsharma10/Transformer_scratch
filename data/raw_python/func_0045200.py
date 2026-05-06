def _log_statistics(self):
        """
        Log statistics about the number of rows and number of rows per second.
        """
        rows_per_second_trans = self._count_total / (self._time1 - self._time0)
        rows_per_second_load = self._count_transform / (self._time2 - self._time1)
        rows_per_second_overall = self._count_total / (self._time3 - self._time0)

        self._log('Number of rows processed            : {0:d}'.format(self._count_total))
        self._log('Number of rows transformed          : {0:d}'.format(self._count_transform))
        self._log('Number of rows ignored              : {0:d}'.format(self._count_ignore))
        self._log('Number of rows parked               : {0:d}'.format(self._count_park))
        self._log('Number of errors                    : {0:d}'.format(self._count_error))
        self._log('Number of rows per second processed : {0:d}'.format(int(rows_per_second_trans)))
        self._log('Number of rows per second loaded    : {0:d}'.format(int(rows_per_second_load)))
        self._log('Number of rows per second overall   : {0:d}'.format(int(rows_per_second_overall)))