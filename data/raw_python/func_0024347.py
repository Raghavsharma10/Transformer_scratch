def seek(self, offset):
        """
        shifts on a given number of record in the original file
        :param offset: number of record
        """
        if self._shifts:
            if 0 <= offset < len(self._shifts):
                current_pos = self._file.tell()
                new_pos = self._shifts[offset]
                if current_pos != new_pos:
                    if current_pos == self._shifts[-1]:  # reached the end of the file
                        self._data = self.__reader()
                        self.__file = iter(self._file.readline, '')
                        self._file.seek(0)
                        next(self._data)
                        if offset:  # move not to the beginning of the file
                            self._file.seek(new_pos)
                    else:
                        if not self.__already_seeked:
                            if self._shifts[0] < current_pos:  # in the middle of the file
                                self._data.send(True)
                            self.__already_seeked = True
                        self._file.seek(new_pos)
            else:
                raise IndexError('invalid offset')
        else:
            raise self._implement_error