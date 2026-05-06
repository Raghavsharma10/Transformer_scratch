def on_packet(self, packet_type):
        """
        Registers a function to be called when packet data is received with a
        specific type.
        """
        def _wrapper(fn):
            return self.callbacks.register(packet_type, fn)
        return _wrapper