def create_pull_from_issue(self, issue, base, head):
        """Create a pull request from issue #``issue``.

        :param int issue: (required), issue number
        :param str base: (required), e.g., 'master'
        :param str head: (required), e.g., 'username:branch'
        :returns: :class:`PullRequest <github3.pulls.PullRequest>` if
            successful, else None
        """
        if int(issue) > 0:
            data = {'issue': issue, 'base': base, 'head': head}
            return self._create_pull(data)
        return None