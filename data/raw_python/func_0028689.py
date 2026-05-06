def get_league(self, slug):
        """
        Returns a Pokemon League object containing the details about the
        league.
        """
        endpoint = '/league/' + slug
        return self.make_request(self.BASE_URL + endpoint)