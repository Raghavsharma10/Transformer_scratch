def get_likes(self, offset=0, limit=50):
        """ Get user's likes. """
        response = self.client.get(
            self.client.USER_LIKES % (self.name, offset, limit))
        return self._parse_response(response, strack)