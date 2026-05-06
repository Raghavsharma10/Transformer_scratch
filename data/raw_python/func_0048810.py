def instruction(self, val):
        """Set the action and command from an instruction"""
        self._instruction = val
        if isinstance(val, tuple):
            if len(val) is 2:
                self._action, self.command = val
            else:
                self._action, self.command, self.extra = val
        else:
            split = val.split(" ", 1)
            if split[0] == "FROM":
                split = val.split(" ", 2)

            if len(split) == 3:
                self._action, self.command, self.extra = split
            else:
                self._action, self.command = split