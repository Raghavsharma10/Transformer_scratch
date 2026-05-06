def get_facets(self):
        '''
        Returns a dictionary of facets::

            >>> res = solr.query('SolrClient_unittest',{
                    'q':'product_name:Lorem',
                    'facet':True,
                    'facet.field':'facet_test',
            })... ... ... ...
            >>> res.get_results_count()
            4
            >>> res.get_facets()
            {'facet_test': {'ipsum': 0, 'sit': 0, 'dolor': 2, 'amet,': 1, 'Lorem': 1}}

        '''
        if not hasattr(self,'facets'):
            self.facets = {}
            data = self.data
            if 'facet_counts' in data.keys() and type(data['facet_counts']) == dict:
                if 'facet_fields' in data['facet_counts'].keys() and type(data['facet_counts']['facet_fields']) == dict:
                    for facetfield in data['facet_counts']['facet_fields']:
                        if type(data['facet_counts']['facet_fields'][facetfield] == list):
                            l = data['facet_counts']['facet_fields'][facetfield]
                            self.facets[facetfield] = OrderedDict(zip(l[::2],l[1::2]))
                return self.facets
            else:
                raise SolrResponseError("No Facet Information in the Response")
        else:
            return self.facets