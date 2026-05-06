def search(self, **kwargs):
        """
        Method to search neighbors based on extends search.

        :param search: Dict containing QuerySets to find neighbors.
        :param include: Array containing fields to include on response.
        :param exclude: Array containing fields to exclude on response.
        :param fields:  Array containing fields to override default fields.
        :param kind: Determine if result will be detailed ('detail') or basic ('basic').
        :return: Dict containing neighbors
        """

        return super(ApiV4Neighbor, self).get(self.prepare_url(
            'api/v4/neighbor/', kwargs))