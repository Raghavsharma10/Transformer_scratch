def iter_milestones(self, state=None, sort=None, direction=None,
                        number=-1, etag=None):
        """Iterates over the milestones on this repository.

        :param str state: (optional), state of the milestones, accepted
            values: ('open', 'closed')
        :param str sort: (optional), how to sort the milestones, accepted
            values: ('due_date', 'completeness')
        :param str direction: (optional), direction to sort the milestones,
            accepted values: ('asc', 'desc')
        :param int number: (optional), number of milestones to return.
            Default: -1 returns all milestones
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of
            :class:`Milestone <github3.issues.milestone.Milestone>`\ s
        """
        url = self._build_url('milestones', base_url=self._api)
        accepted = {'state': ('open', 'closed'),
                    'sort': ('due_date', 'completeness'),
                    'direction': ('asc', 'desc')}
        params = {'state': state, 'sort': sort, 'direction': direction}
        for (k, v) in list(params.items()):
            if not (v and (v in accepted[k])):  # e.g., '' or None
                del params[k]
        if not params:
            params = None
        return self._iter(int(number), url, Milestone, params, etag)