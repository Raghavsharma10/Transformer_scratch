def error(self, message, set_error_state=False):
        """Log an error messsage.

        :param message:  Log message.

        """
        if set_error_state:
            if message not in self._errors:
                self._errors.append(message)

            self.set_error_state()

        self.logger.error(message)