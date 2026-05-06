def message_channel(self, message):
        """
        We won't receive our own messages, so log them manually.
        """
        self.log(None, message)
        super(BaseBot, self).message_channel(message)