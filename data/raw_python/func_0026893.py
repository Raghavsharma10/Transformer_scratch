def on_welcome(self, connection, event):
        """
        Join the channel once connected to the IRC server.
        """
        connection.join(self.channel, key=settings.IRC_CHANNEL_KEY or "")