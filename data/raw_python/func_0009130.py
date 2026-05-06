def read_handshake(self):
        """Read and process an initial handshake message from Storm."""
        msg = self.read_message()
        pid_dir, _conf, _context = msg["pidDir"], msg["conf"], msg["context"]

        # Write a blank PID file out to the pidDir
        open(join(pid_dir, str(self.pid)), "w").close()
        self.send_message({"pid": self.pid})

        return _conf, _context