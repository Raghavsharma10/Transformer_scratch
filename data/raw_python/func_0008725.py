def get_images(self, limit=None):
        """Return all of the images associated with the user."""
        url = (self._imgur._base_url + "/3/account/{0}/"
               "images/{1}".format(self.name, '{}'))
        resp = self._imgur._send_request(url, limit=limit)
        return [Image(img, self._imgur) for img in resp]