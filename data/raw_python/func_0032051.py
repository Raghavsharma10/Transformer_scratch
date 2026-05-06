def parse_message(cls, message):
        """
        Parses a message received from the Pebble. Uses Pebble Protocol framing to figure out what sort of packet
        it is. If the packet is registered (has been defined and imported), returns the deserialised packet, which will
        not necessarily be the same class as this. Otherwise returns ``None``.

        Also returns the length of the message consumed during deserialisation.

        :param message: A serialised message received from the Pebble.
        :type message: bytes
        :return: ``(decoded_message, decoded length)``
        :rtype: (:class:`PebblePacket`, :any:`int`)
        """
        length = struct.unpack_from('!H', message, 0)[0] + 4
        if len(message) < length:
            raise IncompleteMessage()
        command, = struct.unpack_from('!H', message, 2)
        if command in _PacketRegistry:
            return _PacketRegistry[command].parse(message[4:length])[0], length
        else:
            return None, length