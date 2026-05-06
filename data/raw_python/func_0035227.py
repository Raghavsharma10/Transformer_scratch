def pause(self):
        """
        Sends a "play" command to the player.
        """
        msg = cr.Message()
        msg.type = cr.PAUSE
        self.send_message(msg)