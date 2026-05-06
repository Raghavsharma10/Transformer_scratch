def _filter_gameplay_stats(self, data, game_mode, game_region):
        """Returns gameplay stats that are filtered by game_mode and game_region.

            :param data: Json of gameplay stats.
            :type data: dict
            :param game_mode: Target game mode.
            :type game_mode: str
            :param game_region: Target game region.
            :type game_region: str
            :return: return list of gameplay stats with target game mode and region.
            :rtype: list
        """
        return_data = []
        for stat in data['Stats']:
            if self._is_target_game_mode(stat, game_mode) and self._is_target_region(stat, game_region):
                return_data.append(stat)
        return return_data