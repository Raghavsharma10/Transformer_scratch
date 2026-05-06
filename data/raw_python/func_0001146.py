def _send(self, message):
        """Send an email.

        Helper method that does the actual sending.
        """
        if not message.recipients():
            return False
        try:
            self.connection.sendmail(
                message.sender,
                message.recipients(),
                message.message().as_string(),
            )
        except Exception as e:
            logger.error(
                "Error sending a message to server %s:%s: %s",
                self.host,
                self.port,
                e,
            )
            if not self.fail_silently:
                raise
            return False
        return True