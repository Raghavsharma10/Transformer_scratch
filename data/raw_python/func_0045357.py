def set_params(self, **params):
        """
        add/update solr query parameters
        """
        if self._solr_locked:
            raise Exception("Query already executed, no changes can be made."
                            "%s %s" % (self._solr_query, self._solr_params)
                            )
        self._solr_params.update(params)