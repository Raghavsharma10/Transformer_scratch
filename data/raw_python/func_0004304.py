def search(self, **kwargs):
        """
        Method to search ipv6's based on extends search.

        :param search: Dict containing QuerySets to find ipv6's.
        :param include: Array containing fields to include on response.
        :param exclude: Array containing fields to exclude on response.
        :param fields:  Array containing fields to override default fields.
        :param kind: Determine if result will be detailed ('detail') or basic ('basic').
        :return: Dict containing ipv6's
        """

        return super(ApiV4IPv6, self).get(self.prepare_url('api/v4/ipv6/',
                                                           kwargs))