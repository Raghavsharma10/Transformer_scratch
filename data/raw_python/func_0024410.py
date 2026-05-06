def shift_down_right(self, times=1):
        """
        Finds Location shifted down right by 1

        :rtype: Location
        """
        try:
            return Location(self._rank - times, self._file + times)
        except IndexError as e:
            raise IndexError(e)