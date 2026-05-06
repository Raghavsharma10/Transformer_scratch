def __notify(self, sender, content):
        """
        Calls back listener when a message is received
        """
        if self.handle_message is not None:
            try:
                self.handle_message(sender, content)
            except Exception as ex:
                logging.exception("Error calling message listener: %s", ex)