def ensure_connection(self, errback=None, max_retries=None,
            interval_start=2, interval_step=2, interval_max=30):
        """Ensure we have a connection to the server.

        If not retry establishing the connection with the settings
        specified.

        :keyword errback: Optional callback called each time the connection
          can't be established. Arguments provided are the exception
          raised and the interval that will be slept ``(exc, interval)``.

        :keyword max_retries: Maximum number of times to retry.
          If this limit is exceeded the connection error will be re-raised.

        :keyword interval_start: The number of seconds we start sleeping for.
        :keyword interval_step: How many seconds added to the interval
          for each retry.
        :keyword interval_max: Maximum number of seconds to sleep between
          each retry.

        """
        retry_over_time(self.connect, self.connection_errors, (), {},
                        errback, max_retries,
                        interval_start, interval_step, interval_max)
        return self