def write_all(self, data):
        """
        :param list[int] data:
        """
        self._set_cursor_x(0)
        self._set_cursor_y(0)

        self.DC.on()

        self._write(data)