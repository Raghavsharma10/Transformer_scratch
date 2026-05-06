def flush(self):
        """
        Flush the buffer, if applicable.
        """
        if self.__buffer.tell() > 0:
            # Write the buffer to log
            # noinspection PyProtectedMember
            self.__logger._log(level=self.__log_level, msg=self.__buffer.getvalue().strip(),
                               record_filter=StdErrWrapper.__filter_record)
            # Remove the old buffer
            self.__buffer.truncate(0)
            self.__buffer.seek(0)