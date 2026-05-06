def as_dict(self):
        '''Model metadata in a dictionary'''
        pk = self.pk
        id_type = 3
        if pk.type == 'auto':
            id_type = 1
        return {'id_name': pk.name,
                'id_type': id_type,
                'sorted': bool(self.ordering),
                'autoincr': self.ordering and self.ordering.auto,
                'multi_fields': [field.name for field in self.multifields],
                'indices': dict(((idx.attname, idx.unique)
                                 for idx in self.indices))}