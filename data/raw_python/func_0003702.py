def handle(self):
        """
            Performs endless processing of socket input/output, passing
            cooked information onto the local process.
        """
        while True:
            toRead = select.select([self.local, self.remote], [], [], 0.1)[0]
            if self.local in toRead:
                data = os.read(self.local, 4096)
                self.sock.sendall(data)
                continue
            if self.remote in toRead or self.rawq:
                buf = self.read_eager()
                os.write(self.local, buf)
                continue