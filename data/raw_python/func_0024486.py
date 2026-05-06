def close(self, force=False):
        """
        close opened file

        :param force: force closing of externally opened file or buffer
        """
        if self.__write:
            self.write = self.__write_adhoc
            self.__write = False

        if not self._is_buffer or force:
            self._file.close()