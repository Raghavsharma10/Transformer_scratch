def send_packet(self, packet):
        """
        Sends a message to the Pebble.

        :param packet: The message to send.
        :type packet: .PebblePacket
        """
        if self.log_packet_level:
            logger.log(self.log_packet_level, "-> %s", packet)
        serialised = packet.serialise_packet()
        self.event_handler.broadcast_event("raw_outbound", serialised)
        self.send_raw(serialised)