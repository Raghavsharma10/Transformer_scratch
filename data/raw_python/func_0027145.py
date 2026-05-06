def children_rest_names(self):
        """ Gets the list of all possible children ReST names.

            Returns:
                list: list containing all possible rest names as string

            Example:
                >>> entity = NUEntity()
                >>> entity.children_rest_names
                ["foo", "bar"]
        """

        names = []

        for fetcher in self.fetchers:
            names.append(fetcher.__class__.managed_object_rest_name())

        return names