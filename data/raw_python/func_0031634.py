def run_sync(self):
        """
        Runs the message loop until the Pebble disconnects. This method will block until the watch disconnects or
        a fatal error occurs.

        For alternatives that don't block forever, see :meth:`pump_reader` and :meth:`run_async`.
        """
        while self.connected:
            try:
                self.pump_reader()
            except PacketDecodeError as e:
                logger.warning("Packet decode failed: %s", e)
            except ConnectionError:
                break