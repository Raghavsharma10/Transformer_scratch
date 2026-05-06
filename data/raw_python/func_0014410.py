def get_jsonfacet_counts_as_dict(self, field, data=None):
        '''
        EXPERIMENTAL
        Takes facets and returns then as a dictionary that is easier to work with,
        for example, if you are getting something this::

            {'facets': {'count': 50,
              'test': {'buckets': [{'count': 10,
                 'pr': {'buckets': [{'count': 2, 'unique': 1, 'val': 79},
                   {'count': 1, 'unique': 1, 'val': 9}]},
                 'pr_sum': 639.0,
                 'val': 'consectetur'},
                {'count': 8,
                 'pr': {'buckets': [{'count': 1, 'unique': 1, 'val': 9},
                   {'count': 1, 'unique': 1, 'val': 31},
                   {'count': 1, 'unique': 1, 'val': 33}]},
                 'pr_sum': 420.0,
                 'val': 'auctor'},
                {'count': 8,
                 'pr': {'buckets': [{'count': 2, 'unique': 1, 'val': 94},
                   {'count': 1, 'unique': 1, 'val': 25}]},
                 'pr_sum': 501.0,
                 'val': 'nulla'}]}}}


        This should return you something like this::

            {'test': {'auctor': {'count': 8,
                                 'pr': {9: {'count': 1, 'unique': 1},
                                        31: {'count': 1, 'unique': 1},
                                        33: {'count': 1, 'unique': 1}},
                                 'pr_sum': 420.0},
                      'consectetur': {'count': 10,
                                      'pr': {9: {'count': 1, 'unique': 1},
                                             79: {'count': 2, 'unique': 1}},
                                      'pr_sum': 639.0},
                      'nulla': {'count': 8,
                                'pr': {25: {'count': 1, 'unique': 1},
                                       94: {'count': 2, 'unique': 1}},
                                'pr_sum': 501.0}}}
        '''
        data = data if data else self.data['facets']
        if field not in data:
            raise ValueError("Field To start Faceting on not specified.")
        out = { field: self._json_rec_dict(data[field]['buckets']) }
        return out