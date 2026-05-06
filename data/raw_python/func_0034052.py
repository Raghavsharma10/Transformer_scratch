def send_respawn(self):
        """
        Respawns the player.
        """
        nick = self.player.nick
        self.send_struct('<B%iH' % len(nick), 0, *map(ord, nick))