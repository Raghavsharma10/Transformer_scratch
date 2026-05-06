def rnd_date_array(self, size, start=date(1970, 1, 1), end=date.today()):
        """Array or Matrix of random date generator.
        """
        if isinstance(start, string_types):
            start = self.str2date(start)
        if isinstance(end, string_types):
            end = self.str2date(end)
        if start > end:
            raise ValueError("start time has to be earlier than end time")

        return self.randn(size, self._rnd_date, start, end)