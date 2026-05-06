def get_facet_pivot(self):
        '''
        Parses facet pivot response. Example::
            >>> res = solr.query('SolrClient_unittest',{
            'q':'*:*',
            'fq':'price:[50 TO *]',
            'facet':True,
            'facet.pivot':'facet_test,price' #Note how there is no space between fields. They are just separated by commas
            })
            >>> res.get_facet_pivot()
            {'facet_test,price': {'Lorem': {89: 1, 75: 1}, 'ipsum': {53: 1, 70: 1, 55: 1, 89: 1, 74: 1, 93: 1, 79: 1}, 'dolor': {61: 1, 94: 1}, 'sit': {99: 1, 50: 1, 67: 1, 52: 1, 54: 1, 71: 1, 72: 1, 84: 1, 62: 1}, 'amet,': {68: 1}}}

        This method has built in recursion and can support indefinite number of facets. However, note that the output format is significantly massaged since Solr by default outputs a list of fields in each pivot field.
        '''
        if not hasattr(self,'facet_pivot'):
            self.facet_pivot = {}
            if 'facet_counts' in self.data.keys():
                pivots = self.data['facet_counts']['facet_pivot']
                for fieldset in pivots:
                    self.facet_pivot[fieldset] = {}
                    for sub_field_set in pivots[fieldset]:
                        res = self._rec_subfield(sub_field_set)
                        self.facet_pivot[fieldset].update(res)
                return self.facet_pivot
        else:
            return self.facet_pivot