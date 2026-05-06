def comment(self, text):
        """
        Make a top-level comment to this.

        :param text: The comment text.
        """
        url = self._imgur._base_url + "/3/comment"
        payload = {'image_id': self.id, 'comment': text}
        resp = self._imgur._send_request(url, params=payload, needs_auth=True,
                                         method='POST')
        return Comment(resp, imgur=self._imgur, has_fetched=False)