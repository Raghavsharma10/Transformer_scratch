def fetch(self, from_time, until_time=None):
        """
        This method fetch data from the database according to the period
        given

        fetch(path, fromTime, untilTime=None)

        fromTime is an datetime
        untilTime is also an datetime, but defaults to now.

        Returns a tuple of (timeInfo, valueList)
        where timeInfo is itself a tuple of (fromTime, untilTime, step)

        Returns None if no data can be returned
        """
        until_time = until_time or datetime.now()
        time_info, values = whisper.fetch(self.path,
                                          from_time.strftime('%s'),
                                          until_time.strftime('%s'))
        # build up a list of (timestamp, value)
        start_time, end_time, step = time_info
        current = start_time
        times = []
        while current <= end_time:
            times.append(current)
            current += step
        return zip(times, values)