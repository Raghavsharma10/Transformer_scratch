def create_status(self, sha, state, target_url=None, description=None,
                      context='default'):
        """Create a status object on a commit.

        :param str sha: (required), SHA of the commit to create the status on
        :param str state: (required), state of the test; only the following
            are accepted: 'pending', 'success', 'error', 'failure'
        :param str target_url: (optional), URL to associate with this status.
        :param str description: (optional), short description of the status
        :param str context: (optional), A string label to differentiate this
            status from the status of other systems
        :returns: the status created if successful
        :rtype: :class:`~github3.repos.status.Status`
        """
        json = None
        if sha and state:
            data = {'state': state, 'target_url': target_url,
                    'description': description, 'context': context}
            url = self._build_url('statuses', sha, base_url=self._api)
            self._remove_none(data)
            json = self._json(self._post(url, data=data), 201)
        return Status(json) if json else None