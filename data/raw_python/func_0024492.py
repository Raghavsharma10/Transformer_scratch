def close(self, *args, **kwargs):
        """
        write close tag of MRV file and close opened file

        :param force: force closing of externally opened file or buffer
        """
        if not self.__finalized:
            self._file.write('</cml>')
            self.__finalized = True
        super().close(*args, **kwargs)