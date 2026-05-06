def halt(self):
        """
        Halt current endpoint.
        """
        try:
            self._halt()
        except IOError as exc:
            if exc.errno != errno.EBADMSG:
                raise
        else:
            raise ValueError('halt did not return EBADMSG ?')
        self._halted = True