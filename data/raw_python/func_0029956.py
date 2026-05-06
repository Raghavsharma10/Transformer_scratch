def fatal(self, message):
        """Log a fatal messsage and exit.

        :param message:  Log message.

        """

        self.logger.fatal(message)
        sys.stderr.flush()
        if self.exit_on_fatal:
            sys.exit(1)
        else:
            raise FatalError(message)