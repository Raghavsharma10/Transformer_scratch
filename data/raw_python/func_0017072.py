def create_comment(self, body, sha, path=None, position=None, line=1):
        """Create a comment on a commit.

        :param str body: (required), body of the message
        :param str sha: (required), commit id
        :param str path: (optional), relative path of the file to comment
            on
        :param str position: (optional), line index in the diff to comment on
        :param int line: (optional), line number of the file to comment on,
            default: 1
        :returns: :class:`RepoComment <github3.repos.comment.RepoComment>` if
            successful, otherwise None

        """
        json = None
        if body and sha and (line and int(line) > 0):
            data = {'body': body, 'line': line, 'path': path,
                    'position': position}
            self._remove_none(data)
            url = self._build_url('commits', sha, 'comments',
                                  base_url=self._api)
            json = self._json(self._post(url, data=data), 201)
        return RepoComment(json, self) if json else None