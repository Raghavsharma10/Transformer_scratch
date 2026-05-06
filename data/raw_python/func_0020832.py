def poll(self, timeout=None):
        """Wait for an event to occur.

        If `timeout` is given, if specifies the length of time in milliseconds
        which the function will wait for events before returing. If `timeout`
        is omitted, negative or None, the call will block until there is an
        event.

        Returns a list of events. In case no event is pending, an empty list is
        returned.
        """
        if timeout is None:
            timeout = -1

        ret = api.py_aa_async_poll(self.handle, timeout)
        _raise_error_if_negative(ret)

        events = list()
        for event in (POLL_I2C_READ, POLL_I2C_WRITE, POLL_SPI,
                POLL_I2C_MONITOR):
            if ret & event:
                events.append(event)
        return events