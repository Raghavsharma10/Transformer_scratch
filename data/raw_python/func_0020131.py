def order(self, last):
        '''Perform ordering with respect model fields.'''
        desc = last.desc
        field = last.name
        nested = last.nested
        nested_args = []
        while nested:
            meta = nested.model._meta
            nested_args.extend((self.backend.basekey(meta), nested.name))
            last = nested
            nested = nested.nested
        method = 'ALPHA' if last.field.internal_type == 'text' else ''
        if field == last.model._meta.pkname():
            field = ''
        return {'field': field,
                'method': method,
                'desc': desc,
                'nested': nested_args}