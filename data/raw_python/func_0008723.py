def get_gallery_profile(self):
        """Return the users gallery profile."""
        url = (self._imgur._base_url + "/3/account/{0}/"
               "gallery_profile".format(self.name))
        return self._imgur._send_request(url)