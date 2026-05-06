def _inputcooker_getc(self, block=True):
        """Get one character from the raw queue. Optionally blocking.
        Raise EOFError on end of stream. SHOULD ONLY BE CALLED FROM THE
        INPUT COOKER."""
        if self.rawq:
            ret = self.rawq[0]
            self.rawq = self.rawq[1:]
            return ret
        if not block:
            if not self.inputcooker_socket_ready():
                return ''
        ret = self.sock.recv(20)
        self.eof = not(ret)
        self.rawq = self.rawq + ret
        if self.eof:
            raise EOFError
        return self._inputcooker_getc(block)