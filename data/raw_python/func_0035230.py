def next(self):
        """
        Sends a "next" command to the player.
        """
        msg = cr.Message()
        msg.type = cr.NEXT
        self.send_message(msg)