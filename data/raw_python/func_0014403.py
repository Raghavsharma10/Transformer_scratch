def get_facets_ranges(self):
        '''
        Returns query facet ranges ::

            >>> res = solr.query('SolrClient_unittest',{
                'q':'*:*',
                'facet':True,
                'facet.range':'price',
                'facet.range.start':0,
                'facet.range.end':100,
                'facet.range.gap':10
                })
            >>> res.get_facets_ranges()
            {'price': {'80': 9, '10': 5, '50': 3, '20': 7, '90': 3, '70': 4, '60': 7, '0': 3, '40': 5, '30': 4}}

        '''
        if not hasattr(self,'facet_ranges'):
            self.facet_ranges = {}
            data = self.data
            if 'facet_counts' in data.keys() and type(data['facet_counts']) == dict:
                if 'facet_ranges' in data['facet_counts'].keys() and type(data['facet_counts']['facet_ranges']) == dict:
                    for facetfield in data['facet_counts']['facet_ranges']:
                        if type(data['facet_counts']['facet_ranges'][facetfield]['counts']) == list:
                            l = data['facet_counts']['facet_ranges'][facetfield]['counts']
                            self.facet_ranges[facetfield] = OrderedDict(zip(l[::2],l[1::2]))
                    return self.facet_ranges
            else:
                raise SolrResponseError("No Facet Ranges in the Response")
        else:
            return self.facet_ranges