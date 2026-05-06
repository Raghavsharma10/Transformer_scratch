def wscall(self, method, query=None, callback=None):
        """Submit a request on the websocket"""
        if callback is None:
            self.sock.emit(method, query)
        else:
            self.sock.emitack(method, query, callback)