def connect(self):
        """Connect to the unix domain socket, which is passed to us as self.host

        This is in host because the format we use for the unix domain socket is:

        http+unix://%2Fpath%2Fto%2Fsocket.sock

        """
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

            if has_timeout(self.timeout):
                self.sock.settimeout(self.timeout)

            self.sock.connect(unquote(self.host))
        except socket.error as msg:
            if self.sock:
                self.sock.close()
            self.sock = None

            raise socket.error(msg)