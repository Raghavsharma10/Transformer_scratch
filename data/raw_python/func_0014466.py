def set_keepalive(self, interval):
        """
        Turn on/off keepalive packets (default is off).  If this is set, after
        C{interval} seconds without sending any data over the connection, a
        "keepalive" packet will be sent (and ignored by the remote host).  This
        can be useful to keep connections alive over a NAT, for example.

        @param interval: seconds to wait before sending a keepalive packet (or
            0 to disable keepalives).
        @type interval: int
        """
        self.packetizer.set_keepalive(interval,
            lambda x=weakref.proxy(self): x.global_request('keepalive@lag.net', wait=False))