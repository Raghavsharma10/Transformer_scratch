def disconnect(self, *args, **kwargs):
        """
        WebSocket was disconnected - leave the IRC channel.
        """
        quit_message = "%s %s" % (settings.GNOTTY_VERSION_STRING,
                                  settings.GNOTTY_PROJECT_URL)
        self.client.connection.quit(quit_message)
        super(IRCNamespace, self).disconnect(*args, **kwargs)