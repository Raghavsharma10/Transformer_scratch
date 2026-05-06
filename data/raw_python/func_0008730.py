def get_submissions(self, limit=None):
        """Return a list of the images a user has submitted to the gallery."""
        url = (self._imgur._base_url + "/3/account/{0}/submissions/"
               "{1}".format(self.name, '{}'))
        resp = self._imgur._send_request(url, limit=limit)
        return [_get_album_or_image(thing, self._imgur) for thing in resp]