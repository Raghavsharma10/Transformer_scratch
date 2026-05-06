def start_transmit(self, blocking=False):
        """ Start transmit on port.

        :param blocking: True - wait for traffic end, False - return after traffic start.
        """

        self.session.start_transmit(blocking, False, self)