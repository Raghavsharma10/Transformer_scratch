def backend_fields(self, fields):
        '''Return a two elements tuple containing a list
of fields names and a list of field attribute names.'''
        dfields = self.dfields
        processed = set()
        names = []
        atts = []
        pkname = self.pkname()
        for name in fields:
            if name == pkname or name in processed:
                continue
            elif name in dfields:
                processed.add(name)
                field = dfields[name]
                names.append(field.name)
                atts.append(field.attname)
            else:
                bname = name.split(JSPLITTER)[0]
                if bname in dfields:
                    field = dfields[bname]
                    if field.type in ('json object', 'related object'):
                        processed.add(name)
                        names.append(name)
                        atts.append(name)
        return names, atts