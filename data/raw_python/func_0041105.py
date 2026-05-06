def rnd_datetime_array(self,
                           size, start=datetime(1970, 1, 1), end=datetime.now()):
        """Array or Matrix of random datetime generator.
        """
        if isinstance(start, string_types):
            start = self.str2datetime(start)
        if isinstance(end, str):
            end = self.str2datetime(end)
        if start > end:
            raise ValueError("start time has to be earlier than end time")

        return self.randn(size, self._rnd_datetime, start, end)