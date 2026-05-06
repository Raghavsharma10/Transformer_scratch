def get_tracks(self, offset=0, limit=50):
        """ Get user's tracks. """
        response = self.client.get(
            self.client.USER_TRACKS % (self.name, offset, limit))
        return self._parse_response(response, strack)