def raw(self, query):
        """
        make a raw query

        Args:
        query (str): solr query
        \*\*params: solr parameters
        """
        clone = copy.deepcopy(self)
        clone.adapter._pre_compiled_query = query
        clone.adapter.compiled_query = query
        return clone