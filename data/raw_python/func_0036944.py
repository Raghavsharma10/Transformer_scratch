def timestamp(self, value):
        """
        The local time when the message was written.

        Must follow the format 'Mmm DD HH:MM:SS'.  If
        the day of the month is less than 10, then it
        MUST be represented as a space and then the
        number.

        """
        if not self._timestamp_is_valid(value):
            value = self._calculate_current_timestamp()
        self._timestamp = value