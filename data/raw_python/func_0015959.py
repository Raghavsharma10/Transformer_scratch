def _to_milliseconds(self, time):
        """
        Converts a datetime-like object to epoch, in milliseconds
        Timezone-naive datetime objects are assumed to be in UTC

        Parameters
        ----------
        time : dt.datetime | pd.Timestamp | int

        Returns
        -------
        int
            epoch milliseconds
        """
        if isinstance(time, dt.datetime):
            if time.tzinfo is None:
                time = time.replace(tzinfo=pytz.UTC)
            return int(time.timestamp() * 1e3)
        elif isinstance(time, numbers.Number):
            return time
        else:
            raise NotImplementedError("Time format not supported. Use milliseconds since epoch,\
                                        Datetime or Pandas Datetime")