def connect(self):
        """ Connect to the tcp gateway
        Allow for this function to be keypad agnostic
        If keypad value is omitted, then set it to the hex value of 70 which is the recommended value for an external
        device controlling the system (top of pg 3 of cav6.6_rnet_protocol_v1.01.00.pdf). (In fact I don't know under
        what circumstances we would actually want to pass a keypadID at all).
        """

        try:
            self.sock.connect((self._host, self._port))
            _LOGGER.info("Successfully connected to Russound on %s:%s", self._host, self._port)
            return True
        except socket.error as msg:
            _LOGGER.error("Error trying to connect to Russound controller.")
            _LOGGER.error(msg)
            return False