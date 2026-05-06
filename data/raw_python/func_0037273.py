def _stream(self):  # pragma: no cover
        """Runs in a sub-process to perform stream consumption"""
        self.factory.protocol = LiveStream
        self.factory.datasift = {
            'on_open': self._on_open,
            'on_close': self._on_close,
            'on_message': self._on_message,
            'send_message': None
        }
        if self.config.ssl:
            from twisted.internet import ssl
            options = ssl.optionsForClientTLS(hostname=WEBSOCKET_HOST)
            connectWS(self.factory, options)
        else:
            connectWS(self.factory)
        reactor.run()