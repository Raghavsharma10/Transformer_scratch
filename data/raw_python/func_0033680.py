def send(self, packet_type, bulb, packet_fmt, *packet_args):
        """
        Builds and sends a packet to one or more bulbs.
        """
        packet = build_packet(packet_type, self.gateway.mac, bulb,
                              packet_fmt, *packet_args)
        self.logger('>> %s', _bytes(packet))
        self.sender.put(packet)