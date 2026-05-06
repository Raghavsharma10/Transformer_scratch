def content_bytes(self):
        """Return the start and stop needle."""
        get_message = \
            self._communication.needle_positions.get_line_configuration_message
        return get_message(self._line_number)