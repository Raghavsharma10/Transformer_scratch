def get_recent_matches(self, card_type="micro_card"):
        """
        Calling the Recent Matches API.

        Arg:
           card_type: optional, default to micro_card. Accepted values are
           micro_card & summary_card.
        Return:
           json data
        """

        recent_matches_url = self.api_path + "recent_matches/"
        params = {}
        params["card_type"] = card_type
        response = self.get_response(recent_matches_url, params)
        return response