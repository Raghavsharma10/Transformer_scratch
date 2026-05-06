def send_and_read(self, packet, endpoint, timeout=15):
        """
        Sends a packet, then returns the next response received from that endpoint. This method sets up a listener
        before it actually sends the message, avoiding a potential race.

        .. warning::
           Avoid calling this method from an endpoint callback; doing so is likely to lead to deadlock.

        :param packet: The message to send.
        :type packet: .PebblePacket
        :param endpoint: The endpoint to read from
        :type endpoint: .PacketType
        :param timeout: The maximum time to wait before raising :exc:`.TimeoutError`.
        :return: The message read from the endpoint; of the same type as passed to ``endpoint``.
        """
        queue = self.get_endpoint_queue(endpoint)
        self.send_packet(packet)
        try:
            return queue.get(timeout=timeout)
        finally:
            queue.close()