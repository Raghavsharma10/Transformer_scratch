def get_season_stats(self, season_key):
        """
        Calling Season Stats API.

        Arg:
           season_key: key of the season
        Return:
           json data
        """

        season_stats_url = self.api_path + "season/" + season_key + "/stats/"
        response = self.get_response(season_stats_url)
        return response