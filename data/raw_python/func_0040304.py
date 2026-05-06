def cmd_part(self, connection, sender, target, payload):
        """
        Asks the bot to leave a channel
        """
        if payload:
            connection.part(payload)
        else:
            raise ValueError("No channel given")