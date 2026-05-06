def delete(self):
        """Delete the message."""
        url = self._imgur._base_url + "/3/message/{0}".format(self.id)
        return self._imgur._send_request(url, method='DELETE')