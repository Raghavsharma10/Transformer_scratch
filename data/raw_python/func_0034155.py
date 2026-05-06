def connection(self):
        """Returns an stablished connection"""

        if self._connection:
            return self._connection

        self.log.debug('Initializing connection to %s' % (self.bosh_service.
                                                          netloc))
        if self.bosh_service.scheme == 'http':
            Connection = httplib.HTTPConnection
        elif self.bosh_service.scheme == 'https':
            Connection = httplib.HTTPSConnection
        else:
            # TODO: raise proper exception
            raise Exception('Invalid URL scheme %s' % self.bosh_service.scheme)

        self._connection = Connection(self.bosh_service.netloc, timeout=10)
        self.log.debug('Connection initialized')
        # TODO add exceptions handler there (URL not found etc)

        return self._connection