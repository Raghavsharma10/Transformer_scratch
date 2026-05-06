def get_playlists(self, offset=0, limit=50):
        """ Get user's playlists. """
        response = self.client.get(
            self.client.USER_PLAYLISTS % (self.name, offset, limit))
        return self._parse_response(response, splaylist)

        return playlists