def remove(self, ids, **kwargs):
        """
        Method to delete interface by id.
        :param ids: List containing identifiers of interfaces.
        """
        url = build_uri_with_ids('api/v3/interface/%s/', ids)

        return super(ApiInterfaceRequest, self).delete(self.prepare_url(url, kwargs))