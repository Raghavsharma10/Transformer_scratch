def shift_left(self, times=1):
        """
        Finds Location shifted left by 1

        :rtype: Location
        """
        try:
            return Location(self._rank, self._file - times)
        except IndexError as e:
            raise IndexError(e)