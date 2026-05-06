def send(self, *args):
        """
        Send a number of frames.
        """
        for frame in args:
            self.sock.sendall(self.apply_send_hooks(frame, False).pack())