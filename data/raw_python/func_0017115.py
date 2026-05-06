def iter_pulls(self, state=None, head=None, base=None, sort='created',
                   direction='desc', number=-1, etag=None):
        """List pull requests on repository.

        .. versionchanged:: 0.9.0

            - The ``state`` parameter now accepts 'all' in addition to 'open'
              and 'closed'.

            - The ``sort`` parameter was added.

            - The ``direction`` parameter was added.

        :param str state: (optional), accepted values: ('all', 'open',
            'closed')
        :param str head: (optional), filters pulls by head user and branch
            name in the format ``user:ref-name``, e.g., ``seveas:debian``
        :param str base: (optional), filter pulls by base branch name.
            Example: ``develop``.
        :param str sort: (optional), Sort pull requests by ``created``,
            ``updated``, ``popularity``, ``long-running``. Default: 'created'
        :param str direction: (optional), Choose the direction to list pull
            requests. Accepted values: ('desc', 'asc'). Default: 'desc'
        :param int number: (optional), number of pulls to return. Default: -1
            returns all available pull requests
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of
            :class:`PullRequest <github3.pulls.PullRequest>`\ s
        """
        url = self._build_url('pulls', base_url=self._api)
        params = {}
        if state and state.lower() in ('all', 'open', 'closed'):
            params['state'] = state.lower()
        params.update(head=head, base=base, sort=sort, direction=direction)
        self._remove_none(params)
        return self._iter(int(number), url, PullRequest, params, etag)