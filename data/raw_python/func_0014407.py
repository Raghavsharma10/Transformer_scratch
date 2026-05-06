def get_facet_values_as_list(self, field):
        '''
        :param str field: Name of facet field to retrieve values from.

        Returns facet values as list for a given field. Example::

            >>> res = solr.query('SolrClient_unittest',{
                'q':'*:*',
                'facet':'true',
                'facet.field':'facet_test',
            })
            >>> res.get_facet_values_as_list('facet_test')
            [9, 6, 14, 10, 11]
            >>> res.get_facets()
            {'facet_test': {'Lorem': 9, 'ipsum': 6, 'amet,': 14, 'dolor': 10, 'sit': 11}}

        '''
        facets = self.get_facets()
        out = []
        if field in facets.keys():
            for facetfield in facets[field]:
                out.append(facets[field][facetfield])
            return out
        else:
            raise SolrResponseError("No field in facet output")