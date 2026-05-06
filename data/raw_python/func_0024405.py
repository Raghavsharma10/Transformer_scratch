def shift_down(self, times=1):
        """
        Finds Location shifted down by 1

        :rtype: Location
        """
        try:
            return Location(self._rank - times, self._file)
        except IndexError as e:
            raise IndexError(e)