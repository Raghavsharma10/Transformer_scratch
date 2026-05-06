def send_command(self, command):
        """
        Sends a given command to the HAProxy control socket.

        Returns the response from the socket as a string.

        If a known error response (e.g. "Permission denied.") is given then
        the appropriate exception is raised.
        """
        logger.debug("Connecting to socket %s", self.socket_file_path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_file_path)
        except IOError as e:
            if e.errno == errno.ECONNREFUSED:
                logger.error("Connection refused.  Is HAProxy running?")
                return
            else:
                raise

        sock.sendall((command + "\n").encode())

        response = b""
        while True:
            try:
                chunk = sock.recv(SOCKET_BUFFER_SIZE)
                if chunk:
                    response += chunk
                else:
                    break
            except IOError as e:
                if e.errno not in (errno.EAGAIN, errno.EINTR):
                    raise

        sock.close()

        return self.process_command_response(command, response)