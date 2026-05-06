def get_season_player_stats(self, season_key, player_key):
        """
        Calling Season Player Stats API.

        Arg:
           season_key: key of the season
           player_key: key of the player
        Return:
           json data
        """

        season_player_stats_url = self.api_path + "season/" + season_key + "/player/" + player_key + "/stats/"
        response = self.get_response(season_player_stats_url)
        return response