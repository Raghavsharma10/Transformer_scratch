def connection_made(self, transport):
        '''
        override asyncio.Protocol
        '''

        self._connected = True
        self.transport = transport

        self.remote_ip, self.port = transport.get_extra_info('peername')[:2]

        logging.debug(
            'Connection made (address: {} port: {})'
            .format(self.remote_ip, self.port))

        self.auth_future = self.send_package(protomap.CPROTO_REQ_AUTH,
                                             data=(self._username,
                                                   self._password,
                                                   self._dbname),
                                             timeout=10)

        self._password = None
        self.on_connection_made()