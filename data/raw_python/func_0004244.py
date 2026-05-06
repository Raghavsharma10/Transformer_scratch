def search(self, **kwargs):
        """
        Method to search Virtual Interfaces based on extends search.

        :param search: Dict containing QuerySets to find Virtual Interfaces.
        :param include: Array containing fields to include on response.
        :param exclude: Array containing fields to exclude on response.
        :param fields:  Array containing fields to override default fields.
        :param kind: Determine if result will be detailed ('detail') or basic ('basic').
        :return: Dict containing Virtual Interfaces
        """

        return super(ApiV4VirtualInterface, self).get(self.prepare_url(
            'api/v4/virtual-interface/', kwargs))