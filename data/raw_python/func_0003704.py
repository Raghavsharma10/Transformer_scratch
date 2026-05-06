def handle_request(self):
        """
            Handle one request - serve current process to one connection.

            Use close_request() to disconnect this process.
        """
        try:
            request, client_address = self.get_request()
        except socket.error:
            return
        if self.verify_request(request, client_address):
            try:
                # we only serve once, and we want to free up the port
                # for future serves.
                self.socket.close()
                self.process_request(request, client_address)
            except SocketConnected as err:
                self._serve_process(err.slaveFd, err.serverPid)
                return
            except Exception as err:
                self.handle_error(request, client_address)
                self.close_request()