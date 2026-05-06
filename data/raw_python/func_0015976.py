def halt(self, request_type):
        """
        Halt current endpoint.
        """
        try:
            if request_type & ch9.USB_DIR_IN:
                self.read(0)
            else:
                self.write(b'')
        except IOError as exc:
            if exc.errno != errno.EL2HLT:
                raise
        else:
            raise ValueError('halt did not return EL2HLT ?')