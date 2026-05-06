def distinct_values_of(self, field, count_deleted=False):
        """
        Uses riak http search query endpoint for advanced SOLR queries.

        Args:
            field (str): facet field
            count_deleted (bool): ignore deleted or not

        Returns: 
            (dict): pairs of field values and number of counts
        

        """
        solr_params = "facet=true&facet.field=%s&rows=0" % field
        result = self.riak_http_search_query(self.index_name, solr_params, count_deleted)
        facet_fields = result['facet_counts']['facet_fields'][field]
        keys = facet_fields[0::2]
        vals = facet_fields[1::2]

        return dict(zip(keys, vals))