def iter_comments(self, number=-1):
        """Iterate over the comments on this issue.

        :param int number: (optional), number of comments to iterate over
        :returns: iterator of
            :class:`IssueComment <github3.issues.comment.IssueComment>`\ s
        """
        url = self._build_url('comments', base_url=self._api)
        return self._iter(int(number), url, IssueComment)