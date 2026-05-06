def search(self, **kwargs):
        """
        Method to search ipv4's based on extends search.

        :param search: Dict containing QuerySets to find ipv4's.
        :param include: Array containing fields to include on response.
        :param exclude: Array containing fields to exclude on response.
        :param fields:  Array containing fields to override default fields.
        :param kind: Determine if result will be detailed ('detail') or basic ('basic').
        :return: Dict containing ipv4's
        """

        return super(ApiIPv4, self).get(self.prepare_url('api/v3/ipv4/',
                                                         kwargs))