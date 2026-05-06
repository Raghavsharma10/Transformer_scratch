def stop(self):
        """Stop/Close this session

        Close the socket associated with this session and puts Session
        into a state such that it can be re-established later.
        """
        if self.socket is not None:
            self.socket.close()
            self.socket = None
            self.data = None