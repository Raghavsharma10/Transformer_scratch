def previous(self):
        """
        Sends a "previous" command to the player.
        """
        msg = cr.Message()
        msg.type = cr.PREVIOUS
        self.send_message(msg)