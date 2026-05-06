def get_fantasy_credits(self, match_key):
        """
        Calling Fantasy Credit API

        Arg:
            match_key: key of the match
        Return:
            json data
        """

        fantasy_credit_url = self.api_path_v3 + "fantasy-match-credits/" + match_key + "/"
        response = self.get_response(fantasy_credit_url)
        return response