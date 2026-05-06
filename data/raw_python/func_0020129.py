def load_related(self, meta, fname, data, fields, encoding):
        '''Parse data for related objects.'''
        field = meta.dfields[fname]
        if field in meta.multifields:
            fmeta = field.structure_class()._meta
            if fmeta.name in ('hashtable', 'zset'):
                return ((native_str(id, encoding),
                         pairs_to_dict(fdata, encoding)) for
                        id, fdata in data)
            else:
                return ((native_str(id, encoding), fdata) for
                        id, fdata in data)
        else:
            # this is data for stdmodel instances
            return self.build(data, meta, fields, fields, encoding)