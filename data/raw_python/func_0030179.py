def get_formatter(self):
        """Create a fully configured `logging.Formatter`

        Example of formatted log message:
        2017-08-27T20:19:24.424 cpm-example-gew1 progname (23123): hello

        Returns:
            (obj): Instance of `logging.Formatter`
        """
        if not self.fmt:
            self.fmt = ('%(asctime)s.%(msecs)03d {host} {progname} '
                        '(%(process)d): %(message)s').format(
                        host=self.hostname, progname=self.progname)
        if not self.datefmt:
            self.datefmt = '%Y-%m-%dT%H:%M:%S'
        return logging.Formatter(fmt=self.fmt, datefmt=self.datefmt)