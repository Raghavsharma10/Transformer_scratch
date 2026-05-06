def send_request(self, *args, **kwargs):
        """
        Intercept connection errors which suggest that a managed host has
        crashed and raise an exception indicating the location of the log
        """
        try:
            return super(JSHost, self).send_request(*args, **kwargs)
        except RequestsConnectionError as e:
            if (
                self.manager and
                self.has_connected and
                self.logfile and
                'unsafe' not in kwargs
            ):
                raise ProcessError(
                    '{} appears to have crashed, you can inspect the log file at {}'.format(
                        self.get_name(),
                        self.logfile,
                    )
                )
            raise six.reraise(RequestsConnectionError, RequestsConnectionError(*e.args), sys.exc_info()[2])