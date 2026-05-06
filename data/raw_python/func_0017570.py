def get_match(self, match_key, card_type="full_card"):
        """
        Calling the Match API.
    
        Arg:
           match_key: key of the match
           card_type: optional, default to full_card. Accepted values are 
           micro_card, summary_card & full_card.
        Return:
           json data   
        """

        match_url = self.api_path + "match/" + match_key + "/"
        params = {}
        params["card_type"] = card_type
        response = self.get_response(match_url, params)
        return response