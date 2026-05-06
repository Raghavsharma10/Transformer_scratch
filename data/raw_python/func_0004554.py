def get(self, ids, **kwargs):
        """
        Method to get environments vip by their ids

        :param ids: List containing identifiers of environments vip
        :param include: Array containing fields to include on response.
        :param exclude: Array containing fields to exclude on response.
        :param fields: Array containing fields to override default fields.
        :param kind: Determine if result will be detailed ('detail')
                     or basic ('basic').
        :return: Dict containing environments vip
        """
        uri = build_uri_with_ids('api/v3/environment-vip/%s/', ids)
        return super(ApiEnvironmentVip, self).get(
            self.prepare_url(uri, kwargs))