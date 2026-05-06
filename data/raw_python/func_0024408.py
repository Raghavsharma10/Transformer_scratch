def shift_up_right(self, times=1):
        """
        Finds Location shifted up right by 1

        :rtype: Location
        """
        try:
            return Location(self._rank + times, self._file + times)
        except IndexError as e:
            raise IndexError(e)