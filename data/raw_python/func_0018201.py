def streamserver_handle(cls, socket, address):
        '''Translate this class for use in a StreamServer'''
        request = cls.dummy_request()
        request._sock = socket
        server = None
        cls(request, address, server)