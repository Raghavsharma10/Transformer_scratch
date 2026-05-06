def get_album(self, id):
        """Return information about this album."""
        url = self._base_url + "/3/album/{0}".format(id)
        json = self._send_request(url)
        return Album(json, self)