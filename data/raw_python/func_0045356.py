def order_by(self, *args):
        """
        Applies query ordering.

        New parameters are appended to current ones, overwriting existing ones.

        Args:
            **args: Order by fields names.
            Defaults to ascending, prepend with hypen (-) for desecending ordering.


        """
        if self._solr_locked:
            raise Exception("Query already executed, no changes can be made."
                            "%s %s" % (self._solr_query, self._solr_params)
                            )

        for arg in args:
            if arg.startswith('-'):
                self._solr_params['sort'][arg[1:]] = 'desc'
            else:
                self._solr_params['sort'][arg] = 'asc'