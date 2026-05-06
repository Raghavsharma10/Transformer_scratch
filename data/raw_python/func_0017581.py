def get_season_points(self, season_key):
        """
        Calling Season Points API.

        Arg:
           season_key: key of the season
        Return:
           json data
        """

        season_points_url = self.api_path + "season/" + season_key + "/points/"
        response = self.get_response(season_points_url)
        return response