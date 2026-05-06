def right_to_left(self):
        """This is for text that flows Right to Left"""
        self._entry_mode &= ~Command.MODE_INCREMENT
        self.command(self._entry_mode)