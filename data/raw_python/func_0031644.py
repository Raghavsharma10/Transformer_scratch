def send_raw(self, message):
        """
        Sends a raw binary message to the Pebble. No processing will be applied, but any transport framing should be
        omitted.

        :param message: The message to send to the pebble.
        :type message: bytes
        """
        if self.log_protocol_level:
            logger.log(self.log_protocol_level, "-> %s", hexlify(message).decode())
        self.transport.send_packet(message)