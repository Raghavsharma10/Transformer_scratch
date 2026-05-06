def play(self):
        """
        Sends a "play" command to the player.
        """
        msg = cr.Message()
        msg.type = cr.PLAY
        self.send_message(msg)