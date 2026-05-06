def list_all(self, per_page=10, omit=None):
        """List all groups.

        Since the order of groups is determined by recent activity, this is the
        recommended way to obtain a list of all groups. See
        :func:`~groupy.api.groups.Groups.list` for details about ``omit``.

        :param int per_page: number of groups per page
        :param int omit: a comma-separated list of fields to exclude
        :return: a list of groups
        :rtype: :class:`~groupy.pagers.GroupList`
        """
        return self.list(per_page=per_page, omit=omit).autopage()