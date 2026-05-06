def search(self, **kwargs):
        """
        Method to search environments vip based on extends search.

        :param search: Dict containing QuerySets to find environments vip.
        :param include: Array containing fields to include on response.
        :param exclude: Array containing fields to exclude on response.
        :param fields: Array containing fields to override default fields.
        :param kind: Determine if result will be detailed ('detail')
                     or basic ('basic').
        :return: Dict containing environments vip
        """

        return super(ApiEnvironmentVip, self).get(
            self.prepare_url('api/v3/environment-vip/', kwargs))