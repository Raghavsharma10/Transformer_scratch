def send(self, value):
        """
        Send text to stdin. Can only be used on non blocking commands

        Args:
            value (str): the text to write on stdin
        Raises:
            TypeError: If command is blocking
        Returns:
            ShellCommand: return this ShellCommand instance for chaining
        """
        if not self.block and self._stdin is not None:
            self.writer.write("{}\n".format(value))
            return self
        else:
            raise TypeError(NON_BLOCKING_ERROR_MESSAGE)