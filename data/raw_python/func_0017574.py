def get_recent_season_matches(self, season_key):
        """
        Calling specific season recent matches.

        Arg:
           season_key: key of the season.
        Return:
           json date
        """

        season_recent_matches_url = self.api_path + "season/" + season_key + "/recent_matches/"
        response = self.get_response(season_recent_matches_url)
        return response