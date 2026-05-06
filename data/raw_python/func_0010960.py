def level(self):
        """
        Returns the the user's profile level, note that this runs a separate
        request because the profile level data isn't in the standard player summary
        output even though it should be. Which is also why it's not implemented
        as a separate class. You won't need this output and not the profile output
        """

        level_key = "player_level"

        if level_key in self._api["response"]:
            return self._api["response"][level_key]

        try:
            lvl = api.interface("IPlayerService").GetSteamLevel(steamid=self.id64)["response"][level_key]
            self._api["response"][level_key] = lvl
            return lvl
        except:
            return -1