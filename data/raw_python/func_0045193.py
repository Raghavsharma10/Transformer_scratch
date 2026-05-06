def _handle_exception(self, row, exception):
        """
        Logs an exception occurred during transformation of a row.

        :param list|dict|() row: The source row.
        :param Exception exception: The exception.
        """
        self._log('Error during processing of line {0:d}.'.format(self._source_reader.row_number))
        self._log(row)
        self._log(str(exception))
        self._log(traceback.format_exc())