def player_mode_stats(self, player_handle, game_mode=constants.GAME_MODE_WILDCARD, game_region=constants.GAME_REGION_WILDCARD):
        """Returns the stats for a particular mode of play,
        accepts solo, duo and squad.  Will return both regional
        and global stats.  Default gamemode is solo
        by Zac: Add parameter game_region to extract player stats by region directly
        """
        if game_mode not in constants.GAME_MODES:
            raise APIException("game_mode must be one of: solo, duo, squad, all")
        if game_region not in constants.GAME_REGIONS:
            raise APIException("game_region must be one of: as, na, agg, sea, eu, oc, sa, all")
        try:
            data = self._get_player_profile(player_handle)
            data = self._filter_gameplay_stats(data, game_mode, game_region)
            return data
        except BaseException as error:
            print('Unhandled exception: ' + str(error))
            raise