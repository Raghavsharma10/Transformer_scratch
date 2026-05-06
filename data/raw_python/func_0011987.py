def connect(self, sock):
        """Attach a given socket to a channel"""
        def cbwrap(*args, **kwargs):
            """Callback wrapper; passes in response_type"""
            self.callback(self.response_type, *args, **kwargs)

        self.sock = sock
        self.sock.subscribe(self.channel)
        self.sock.onchannel(self.channel, cbwrap)