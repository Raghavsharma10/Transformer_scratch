def iter_org_issues(self, name, filter='', state='', labels='', sort='',
                        direction='', since=None, number=-1, etag=None):
        """Iterate over the organnization's issues if the authenticated user
        belongs to it.

        :param str name: (required), name of the organization
        :param str filter: accepted values:
            ('assigned', 'created', 'mentioned', 'subscribed')
            api-default: 'assigned'
        :param str state: accepted values: ('open', 'closed')
            api-default: 'open'
        :param str labels: comma-separated list of label names, e.g.,
            'bug,ui,@high'
        :param str sort: accepted values: ('created', 'updated', 'comments')
            api-default: created
        :param str direction: accepted values: ('asc', 'desc')
            api-default: desc
        :param since: (optional), Only issues after this date will
            be returned. This can be a `datetime` or an ISO8601 formatted
            date string, e.g., 2012-05-20T23:10:27Z
        :type since: datetime or string
        :param int number: (optional), number of issues to return. Default:
            -1, returns all available issues
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Issue <github3.issues.Issue>`
        """
        url = self._build_url('orgs', name, 'issues')
        # issue_params will handle the since parameter
        params = issue_params(filter, state, labels, sort, direction, since)
        return self._iter(int(number), url, Issue, params, etag)