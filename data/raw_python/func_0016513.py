def close(self):
        """Close the channel if open."""
        if self._channel and not self._channel.handler.channel_close:
            self._channel.close()
        self._channel_ref = None