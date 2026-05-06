def tell(self):
        """
        :return: number of records processed from the original file
        """
        if self._shifts:
            t = self._file.tell()
            if t == self._shifts[0]:
                return 0
            elif t == self._shifts[-1]:
                return len(self._shifts) - 1
            elif t in self._shifts:
                return bisect_left(self._shifts, t)
            else:
                return bisect_left(self._shifts, t) - 1
        raise self._implement_error