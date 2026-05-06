def _is_target_game_mode(self, stat, game_mode):
        """Returns if the stat matches target game mode.

            :param stat: Json of gameplay stat.
            :type stat: dict
            :param game_mode: Target game mode.
            :type game_mode: str
            :return: return does the stat matches target game mode.
            :rtype: bool
        """
        if game_mode == constants.GAME_MODE_WILDCARD:
            return True
        return stat['Match'] == game_mode