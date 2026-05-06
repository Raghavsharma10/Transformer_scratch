def get(self, ids, **kwargs):
        """
        Method to get interfaces by their ids.
        :param ids: List containing identifiers of interfaces.
        :return: Dict containing interfaces.
        """

        url = build_uri_with_ids('api/v3/interface/%s/', ids)

        return super(ApiInterfaceRequest, self).get(self.prepare_url(url, kwargs))