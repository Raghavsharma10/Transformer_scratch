def _is_target_region(self, stat, game_region):
        """Returns if the stat matches target game region.

            :param stat: Json of gameplay stat.
            :type stat: dict
            :param game_region: Target game region.
            :type game_region: str
            :return: return does the stat matches target game region.
            :rtype: bool
        """
        if game_region == constants.GAME_REGION_WILDCARD:
            return True
        return stat['Region'] == game_region