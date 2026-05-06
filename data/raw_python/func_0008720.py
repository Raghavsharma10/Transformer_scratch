def get_albums(self, limit=None):
        """
        Return  a list of the user's albums.

        Secret and hidden albums are only returned if this is the logged-in
        user.
        """
        url = (self._imgur._base_url + "/3/account/{0}/albums/{1}".format(self.name,
                                                                       '{}'))
        resp = self._imgur._send_request(url, limit=limit)
        return [Album(alb, self._imgur, False) for alb in resp]