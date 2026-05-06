def reply(self, body):
        """Reply to this review comment with a new review comment.

        :param str body: The text of the comment.
        :returns: The created review comment.
        :rtype: :class:`~github3.pulls.ReviewComment`
        """
        url = self._build_url('comments', base_url=self.pull_request_url)
        index = self._api.rfind('/') + 1
        in_reply_to = self._api[index:]
        json = self._json(self._post(url, data={
            'body': body, 'in_reply_to': in_reply_to
        }), 201)
        return ReviewComment(json, self) if json else None