def initialize(self):
        '''Create or update indices and mappings'''
        if es.indices.exists(self.index_name):
            es.indices.put_mapping(index=self.index_name, doc_type=DOCTYPE, body=MAPPING)
        else:
            es.indices.create(self.index_name, {
                'mappings': {'advice': MAPPING},
                'settings': {'analysis': ANALSYS},
            })