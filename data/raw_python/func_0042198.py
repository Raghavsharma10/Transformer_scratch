def waiting(self, timeout=0):
        "Return True if data is ready for the client."
        if self.linebuffer:
            return True
        (winput, woutput, wexceptions) = select.select((self.sock,), (), (), timeout)
        return winput != []