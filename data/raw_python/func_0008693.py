def update(self, title=None, description=None):
        """Update the image with a new title and/or description."""
        url = (self._imgur._base_url + "/3/image/"
               "{0}".format(self._delete_or_id_hash))
        is_updated = self._imgur._send_request(url, params=locals(),
                                               method='POST')
        if is_updated:
            self.title = title or self.title
            self.description = description or self.description
        return is_updated