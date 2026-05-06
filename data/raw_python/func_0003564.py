def receive_message(self):
        """Receive a message from the file."""
        with self.lock:
            assert self.can_receive_messages()
            message_type = self._read_message_type(self._file)
            message = message_type(self._file, self)
            self._message_received(message)