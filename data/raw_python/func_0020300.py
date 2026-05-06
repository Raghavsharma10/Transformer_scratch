def loadedfields(self):
        '''Generator of fields loaded from database'''
        if self._loadedfields is None:
            for field in self._meta.scalarfields:
                yield field
        else:
            fields = self._meta.dfields
            processed = set()
            for name in self._loadedfields:
                if name in processed:
                    continue
                if name in fields:
                    processed.add(name)
                    yield fields[name]
                else:
                    name = name.split(JSPLITTER)[0]
                    if name in fields and name not in processed:
                        field = fields[name]
                        if field.type == 'json object':
                            processed.add(name)
                            yield field