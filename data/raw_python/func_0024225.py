def tell(self):
        """
        :return: number of records processed from the original file
        """
        if self._shifts:
            t = self._file.tell()
            return bisect_left(self._shifts, t)
        raise self._implement_error