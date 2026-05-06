def get_fantasy_points(self, match_key):
        """
        Calling Fantasy Points API

        Arg:
            match_key: key of the match
        Return:
            json data
        """

        fantasy_points_url = self.api_path_v3 + "fantasy-match-points/" + match_key + "/"
        response = self.get_response(fantasy_points_url)
        return response