def query(self):
        """Generate a new query for CDMRemote.

        This handles turning on compression if necessary.

        Returns
        -------
        HTTPQuery
            The created query.

        """
        q = super(CDMRemote, self).query()

        # Turn on compression if it's been set on the object
        if self.deflate:
            q.add_query_parameter(deflate=self.deflate)

        return q