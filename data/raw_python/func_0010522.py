def finish(self, *args, **kwargs):
        """
        Call this function to inform that the operation is finished.
        :param args: extra positional arguments to pass on
        :param kwargs: extra keyword arguments to pass on
        """
        log.debug('finish(args={args}, kwargs={kwargs})'.format(args=args, kwargs=kwargs))
        self.on_finish(*args, **kwargs)