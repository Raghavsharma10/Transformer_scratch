def all(self, query=None, **kwargs):
        """
        Gets all assets of a space.
        """

        if query is None:
            query = {}

        normalize_select(query)

        return super(AssetsProxy, self).all(query, **kwargs)