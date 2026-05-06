def message(self, msg):
        """Send a message to third party applications
        """
        for broker in self.message_brokers:
            try:
                broker(msg)
            except Exception as exc:
                utils.error(exc)