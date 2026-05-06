def initialize(self, timeouts):
        """ Bind or connect the nanomsg socket to some address """

        # Bind or connect to address
        if self.bind is True:
            self.socket.bind(self.address)
        else:
            self.socket.connect(self.address)

        # Set send and recv timeouts
        self._set_timeouts(timeouts)