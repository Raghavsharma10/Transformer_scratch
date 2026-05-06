def find(self, asset_id, query=None, **kwargs):
        """
        Gets a single asset by ID.
        """

        if query is None:
            query = {}

        normalize_select(query)

        return super(AssetsProxy, self).find(asset_id, query=query, **kwargs)