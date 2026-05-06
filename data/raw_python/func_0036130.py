def receive_until_end(self, timeout=None):
        """
        Reads and blocks until the socket closes

        Used for the "shell" command, where STDOUT and STDERR
        are just redirected to the terminal with no length
        """
        if self.receive_fixed_length(4) != "OKAY":
            raise SocketError("Socket communication failed: "
                              "the server did not return a valid response")

        # The time at which the receive starts
        start_time = time.clock()

        output = ""

        while True:
            if timeout is not None:
                self.socket.settimeout(timeout - (time.clock() - start_time))
            
            chunk = ''
            try:
                chunk = self.socket.recv(4096).decode("ascii")
            except socket.timeout:
                return output            

            if not chunk:
                return output

            output += chunk