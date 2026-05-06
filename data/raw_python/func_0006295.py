def connection_made(self, transport):
        '''
        override _SiriDBProtocol
        '''

        self.transport = transport
        self.remote_ip, self.port = transport.get_extra_info('peername')[:2]

        logging.debug(
            'Connection made (address: {} port: {})'
            .format(self.remote_ip, self.port))

        self.future = self.send_package(
                protomap.CPROTO_REQ_INFO,
                data=None,
                timeout=10)