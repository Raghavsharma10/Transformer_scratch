def __set_timestamp(self, clock):
        """
        If "clock" is None, set the time now.
        This function is called self.__init__()
        """
        if clock is None:
            unix_timestamp = time.mktime(
                datetime.datetime.now().utctimetuple()
            )
            timestamp = int(unix_timestamp)

            return timestamp

        else:
            return clock