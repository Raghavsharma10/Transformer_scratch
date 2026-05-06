def stop(self):
        """
        Sends a "play" command to the player.
        """
        msg = cr.Message()
        msg.type = cr.STOP
        self.send_message(msg)