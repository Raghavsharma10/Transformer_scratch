def get_comment(self, id):
        """Return information about this comment."""
        url = self._base_url + "/3/comment/{0}".format(id)
        json = self._send_request(url)
        return Comment(json, self)