def send_key(self, key):
        """Sends a key."""
        _LOGGER.info('Queueing key %s', key)
        frame = self._get_key_event_frame(key)

        # Queue it to send immediately following the reception
        # of a keep-alive packet in an attempt to avoid bus collisions.
        self._send_queue.put({'frame': frame})