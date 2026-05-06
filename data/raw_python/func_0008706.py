def get_image(self, id):
        """Return a Image object representing the image with the given id."""
        url = self._base_url + "/3/image/{0}".format(id)
        resp = self._send_request(url)
        return Image(resp, self)