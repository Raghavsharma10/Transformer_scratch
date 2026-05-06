def current_game(self):
        """
        Returns a tuple of 3 elements (each of which may be None if not available):
        Current game app ID, server ip:port, misc. extra info (eg. game title)
        """
        obj = self._prof
        gameid = obj.get("gameid")
        gameserverip = obj.get("gameserverip")
        gameextrainfo = obj.get("gameextrainfo")
        return (int(gameid) if gameid else None, gameserverip, gameextrainfo)