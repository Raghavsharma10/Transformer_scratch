def related_lua_args(self):
        '''Generator of load_related arguments'''
        related = self.queryelem.select_related
        if related:
            meta = self.meta
            for rel in related:
                field = meta.dfields[rel]
                relmodel = field.relmodel
                bk = self.backend.basekey(relmodel._meta) if relmodel else ''
                fields = list(related[rel])
                if meta.pkname() in fields:
                    fields.remove(meta.pkname())
                    if not fields:
                        fields.append('')
                ftype = field.type if field in meta.multifields else ''
                data = {'field': field.attname, 'type': ftype,
                        'bk': bk, 'fields': fields}
                yield field.name, data