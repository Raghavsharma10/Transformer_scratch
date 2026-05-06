def target(self, hosts):
        """Temporarily retarget the client for one call.  This is useful
        when having to deal with a subset of hosts for one call.
        """
        if self.__is_retargeted:
            raise TypeError('Cannot use target more than once.')
        rv = FanoutClient(hosts, connection_pool=self.connection_pool,
                          max_concurrency=self._max_concurrency)
        rv._cb_poll = self._cb_poll
        rv.__is_retargeted = True
        return rv