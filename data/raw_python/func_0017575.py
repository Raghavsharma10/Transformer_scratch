def get_recent_seasons(self):
        """
        Calling the Recent Season API.

        Return:
           json data
        """

        recent_seasons_url = self.api_path + "recent_seasons/"
        response = self.get_response(recent_seasons_url)
        return response