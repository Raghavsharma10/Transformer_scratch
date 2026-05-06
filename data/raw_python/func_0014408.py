def get_facet_keys_as_list(self,field):
        '''
        :param str field: Name of facet field to retrieve keys from.

        Similar to get_facet_values_as_list but returns the list of keys as a list instead.
        Example::

            >>> r.get_facet_keys_as_list('facet_test')
            ['Lorem', 'ipsum', 'amet,', 'dolor', 'sit']

        '''
        facets = self.get_facets()
        if facets == -1:
            return facets
        if field in facets.keys():
            return [x for x in facets[field]]