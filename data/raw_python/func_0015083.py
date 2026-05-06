def _recv(self):
        """
        Receives and returns a message from Scratch
        """
        prefix = self._read(self.prefix_len)
        msg = self._read(self._extract_len(prefix))
        return prefix + msg