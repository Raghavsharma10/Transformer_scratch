def remove_from_gallery(self):
        """Remove this image from the gallery."""
        url = self._imgur._base_url + "/3/gallery/{0}".format(self.id)
        self._imgur._send_request(url, needs_auth=True, method='DELETE')
        if isinstance(self, Image):
            item = self._imgur.get_image(self.id)
        else:
            item = self._imgur.get_album(self.id)
        _change_object(self, item)
        return self