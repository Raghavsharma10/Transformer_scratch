def comment(self, id_num):
        """Get a single comment by its id.

        The catch here is that id is NOT a simple number to obtain. If
        you were to look at the comments on issue #15 in
        sigmavirus24/Todo.txt-python, the first comment's id is 4150787.

        :param int id_num: (required), comment id, see example above
        :returns: :class:`IssueComment <github3.issues.comment.IssueComment>`
        """
        json = None
        if int(id_num) > 0:  # Might as well check that it's positive
            owner, repo = self.repository
            url = self._build_url('repos', owner, repo, 'issues', 'comments',
                                  str(id_num))
            json = self._json(self._get(url), 200)
        return IssueComment(json) if json else None