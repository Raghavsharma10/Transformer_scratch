def kill(self):
        """
        Kill the current non blocking command

        Raises:
            TypeError: If command is blocking
        """
        if self.block:
            raise TypeError(NON_BLOCKING_ERROR_MESSAGE)

        try:
            self.process.kill()
        except ProcessLookupError as exc:
            self.logger.debug(exc)