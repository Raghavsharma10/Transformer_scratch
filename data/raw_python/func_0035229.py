def playpause(self):
        """
        Sends a "playpause" command to the player.
        """
        msg = cr.Message()
        msg.type = cr.PLAYPAUSE
        self.send_message(msg)