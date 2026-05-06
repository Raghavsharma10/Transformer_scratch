def list(self, page=1, per_page=10, omit=None):
        """List groups by page.

        The API allows certain fields to be excluded from the results so that
        very large groups can be fetched without exceeding the maximum
        response size. At the time of this writing, only 'memberships' is
        supported.

        :param int page: page number
        :param int per_page: number of groups per page
        :param int omit: a comma-separated list of fields to exclude
        :return: a list of groups
        :rtype: :class:`~groupy.pagers.GroupList`
        """
        return pagers.GroupList(self, self._raw_list, page=page,
                                per_page=per_page, omit=omit)