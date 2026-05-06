def cmd_join(self, connection, sender, target, payload):
        """
        Asks the bot to join a channel
        """
        if payload:
            connection.join(payload)
        else:
            raise ValueError("No channel given")