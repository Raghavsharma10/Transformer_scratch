def get_season(self, season_key, card_type="micro_card"):
        """
        Calling Season API.

        Arg:
           season_key: key of the season
           card_type: optional, default to micro_card. Accepted values are 
           micro_card & summary_card 
        Return:
           json data
        """

        season_url = self.api_path + "season/" + season_key + "/"
        params = {}
        params["card_type"] = card_type
        response = self.get_response(season_url, params)
        return response