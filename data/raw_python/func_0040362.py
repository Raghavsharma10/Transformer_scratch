def _date_val(self, dt: datetime) -> None:
        """
        Add a date value
        :param dt: datetime to add
        """
        self._tval_char = dt.strftime('%Y-%m-%d %H:%M')
        self._nval_num = (dt.year * 10000) + (dt.month * 100) + dt.day + \
                         (((dt.hour / 100.0) + (dt.minute / 10000.0)) if isinstance(dt, datetime) else 0)