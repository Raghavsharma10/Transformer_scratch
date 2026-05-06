def commit_comment(self, comment_id):
        """Get a single commit comment.

        :param int comment_id: (required), id of the comment used by GitHub
        :returns: :class:`RepoComment <github3.repos.comment.RepoComment>` if
            successful, otherwise None
        """
        url = self._build_url('comments', str(comment_id), base_url=self._api)
        json = self._json(self._get(url), 200)
        return RepoComment(json, self) if json else None