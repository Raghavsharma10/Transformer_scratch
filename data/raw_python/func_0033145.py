def index(self, fields, name=None, **kwargs):
        '''
        Build a new index on a cube.

        Examples:
            + index('field_name')

        :param fields: A single field or a list of (key, direction) pairs
        :param name: (optional) Custom name to use for this index
        :param collection: cube name
        :param owner: username of cube owner
        '''
        return self.proxy.index(fields=fields, name=name, table=self.name,
                                **kwargs)