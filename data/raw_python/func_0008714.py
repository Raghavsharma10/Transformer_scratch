def search_gallery(self, q):
        """Search the gallery with the given query string."""
        url = self._base_url + "/3/gallery/search?q={0}".format(q)
        resp = self._send_request(url)
        return [_get_album_or_image(thing, self) for thing in resp]