def serialise_packet(self):
        """
        Serialise a message, including framing information inferred from the ``Meta`` inner class of the packet.
        ``self.Meta.endpoint`` must be defined to call this method.

        :return: A serialised message, ready to be sent to the Pebble.
        """
        if not hasattr(self, '_Meta'):
            raise ReferenceError("Can't serialise a packet that doesn't have an endpoint ID.")
        serialised = self.serialise()
        return struct.pack('!HH', len(serialised), self._Meta['endpoint']) + serialised