def left_to_right(self):
        """This is for text that flows Left to Right"""
        self._entry_mode |= Command.MODE_INCREMENT
        self.command(self._entry_mode)