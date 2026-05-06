def send_message(self, message):
        """Send a message to Storm via stdout."""
        if not isinstance(message, dict):
            logger = self.logger if self.logger else log
            logger.error(
                "%s.%d attempted to send a non dict message to Storm: " "%r",
                self.component_name,
                self.pid,
                message,
            )
            return
        self.serializer.send_message(message)