def streamserver_handle(cls, sock, address):
        '''Translate this class for use in a StreamServer'''
        request = cls.false_request()
        request._sock = sock
        server = None
        log.debug("Accepted connection, starting telnet session.")
        try:
            cls(request, address, server)
        except socket.error:
            pass