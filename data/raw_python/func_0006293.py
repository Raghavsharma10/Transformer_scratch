def connection_lost(self, exc):
        '''
        override asyncio.Protocol
        '''
        self._connected = False

        logging.debug(
            'Connection lost (address: {} port: {})'
            .format(self.remote_ip, self.port))

        for pid, (future, task) in self._requests.items():
            task.cancel()
            if future.cancelled():
                continue
            future.set_exception(ConnectionError(
                'Connection is lost before we had an answer on package id: {}.'
                .format(pid)))

        self.on_connection_lost(exc)