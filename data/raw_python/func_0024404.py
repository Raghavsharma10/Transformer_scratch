def shift_up(self, times=1):
        """
        Finds Location shifted up by 1

        :rtype: Location
        """
        try:
            return Location(self._rank + times, self._file)
        except IndexError as e:
            raise IndexError(e)