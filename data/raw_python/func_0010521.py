def update(self, value, *args, **kwargs):
        """
        Call this function to inform that an update is available.
        This function does NOT call finish when value == maximum.
        :param value: The current index/position of the action. (Should be, but must not be, in the range [min, max])
        :param args: extra positional arguments to pass on
        :param kwargs: extra keyword arguments to pass on
        """
        log.debug('update(value={value}, args={args}, kwargs={kwargs})'.format(value=value, args=args, kwargs=kwargs))
        self.on_update(value, *args, **kwargs)