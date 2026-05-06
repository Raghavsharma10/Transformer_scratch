def get_favorites(self):
        """Return the users favorited images."""
        url = self._imgur._base_url + "/3/account/{0}/favorites".format(self.name)
        resp = self._imgur._send_request(url, needs_auth=True)
        return [_get_album_or_image(thing, self._imgur) for thing in resp]