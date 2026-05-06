def search(self, **kwargs):
        """
        Method to search asns based on extends search.

        :param search: Dict containing QuerySets to find asns.
        :param include: Array containing fields to include on response.
        :param exclude: Array containing fields to exclude on response.
        :param fields:  Array containing fields to override default fields.
        :param kind: Determine if result will be detailed ('detail') or basic ('basic').
        :return: Dict containing asns
        """

        return super(ApiV4As, self).get(self.prepare_url(
            'api/v4/as/', kwargs))