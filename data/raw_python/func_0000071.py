def accept(self):
        """
        Start accepting synchronous, asynchronous and service payloads

        Since services are globally defined, only one :py:class:`ServiceRunner`
        may :py:meth:`accept` payloads at any time.
        """
        if self._meta_runner:
            raise RuntimeError('payloads scheduled for %s before being started' % self)
        self._must_shutdown = False
        self._logger.info('%s starting', self.__class__.__name__)
        # force collecting objects so that defunct, migrated and overwritten services are destroyed now
        gc.collect()
        self._adopt_services()
        self.adopt(self._accept_services, flavour=trio)
        self._meta_runner.run()