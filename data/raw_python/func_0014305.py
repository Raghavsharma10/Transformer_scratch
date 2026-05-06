def find(self, entry_id, query=None):
        """
        Gets a single entry by ID.
        """

        if query is None:
            query = {}

        if self.content_type_id is not None:
            query['content_type'] = self.content_type_id

        normalize_select(query)

        return super(EntriesProxy, self).find(entry_id, query=query)