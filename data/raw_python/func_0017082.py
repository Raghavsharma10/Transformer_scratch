def create_pull(self, title, base, head, body=None):
        """Create a pull request of ``head`` onto ``base`` branch in this repo.

        :param str title: (required)
        :param str base: (required), e.g., 'master'
        :param str head: (required), e.g., 'username:branch'
        :param str body: (optional), markdown formatted description
        :returns: :class:`PullRequest <github3.pulls.PullRequest>` if
            successful, else None
        """
        data = {'title': title, 'body': body, 'base': base,
                'head': head}
        return self._create_pull(data)