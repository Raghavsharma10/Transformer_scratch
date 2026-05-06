def create_review_comment(self, body, commit_id, path, position):
        """Create a review comment on this pull request.

        All parameters are required by the GitHub API.

        :param str body: The comment text itself
        :param str commit_id: The SHA of the commit to comment on
        :param str path: The relative path of the file to comment on
        :param int position: The line index in the diff to comment on.
        :returns: The created review comment.
        :rtype: :class:`~github3.pulls.ReviewComment`
        """
        url = self._build_url('comments', base_url=self._api)
        data = {'body': body, 'commit_id': commit_id, 'path': path,
                'position': int(position)}
        json = self._json(self._post(url, data=data), 201)
        return ReviewComment(json, self) if json else None