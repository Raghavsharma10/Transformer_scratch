def time(self):
        """Retrieve the current time from the redis server.

        :rtype: float
        :raises: :exc:`~tredis.exceptions.RedisError`

        """

        def format_response(value):
            """Format a TIME response into a datetime.datetime

            :param list value: TIME response is a list of the number
                of seconds since the epoch and the number of micros
                as two byte strings
            :rtype: float

            """
            seconds, micros = value
            return float(seconds) + (float(micros) / 1000000.0)

        return self._execute([b'TIME'], format_callback=format_response)