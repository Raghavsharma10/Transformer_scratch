def _parse_broadcast(self, msg):
        """
        Given a broacast message, returns the message that was broadcast.
        """
        # get message, remove surrounding quotes, and unescape
        return self._unescape(self._get_type(msg[self.broadcast_prefix_len:]))