def create_comment(self, body):
        """Create a comment on this issue.

        :param str body: (required), comment body
        :returns: :class:`IssueComment <github3.issues.comment.IssueComment>`
        """
        json = None
        if body:
            url = self._build_url('comments', base_url=self._api)
            json = self._json(self._post(url, data={'body': body}),
                              201)
        return IssueComment(json, self) if json else None