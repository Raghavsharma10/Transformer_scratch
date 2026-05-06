def flush(self):
        """
        Sends buffered data to the target
        """
        # Flush buffer
        line = self._buffer.getvalue()
        self._buffer = StringIO()

        # Send the message
        content = {"session_id": self._session, "text": line}
        self._herald.fire(self._peer, beans.Message(MSG_CLIENT_PRINT, content))