def player_skill(self, player_handle, game_mode='solo'):
        """Returns the current skill rating of the player for a specified gamemode,
        default gamemode is solo"""
        if game_mode not in constants.GAME_MODES:
            raise APIException("game_mode must be one of: solo, duo, squad, all")
        try:
            data = self._get_player_profile(player_handle)
            player_stats = {}
            return_data = []
            for stat in data['Stats']:
                if stat['Match'] == game_mode:
                    for datas in stat['Stats']:
                        if datas['label'] == 'Rating':
                            player_stats[stat['Region']] = datas['value']
            return player_stats
        except BaseException as error:
            print('Unhandled exception: ' + str(error))
            raise