def iter_issues(self,
                    milestone=None,
                    state=None,
                    assignee=None,
                    mentioned=None,
                    labels=None,
                    sort=None,
                    direction=None,
                    since=None,
                    number=-1,
                    etag=None):
        """Iterate over issues on this repo based upon parameters passed.

        .. versionchanged:: 0.9.0

            The ``state`` parameter now accepts 'all' in addition to 'open'
            and 'closed'.

        :param int milestone: (optional), 'none', or '*'
        :param str state: (optional), accepted values: ('all', 'open',
            'closed')
        :param str assignee: (optional), 'none', '*', or login name
        :param str mentioned: (optional), user's login name
        :param str labels: (optional), comma-separated list of labels, e.g.
            'bug,ui,@high'
        :param sort: (optional), accepted values:
            ('created', 'updated', 'comments', 'created')
        :param str direction: (optional), accepted values: ('asc', 'desc')
        :param since: (optional), Only issues after this date will
            be returned. This can be a `datetime` or an `ISO8601` formatted
            date string, e.g., 2012-05-20T23:10:27Z
        :type since: datetime or string
        :param int number: (optional), Number of issues to return.
            By default all issues are returned
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Issue <github3.issues.issue.Issue>`\ s
        """
        url = self._build_url('issues', base_url=self._api)

        params = {'assignee': assignee, 'mentioned': mentioned}
        if milestone in ('*', 'none') or isinstance(milestone, int):
            params['milestone'] = milestone
        self._remove_none(params)
        params.update(
            issue_params(None, state, labels, sort, direction,
                         since)
        )

        return self._iter(int(number), url, Issue, params, etag)