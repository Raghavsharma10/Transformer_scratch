def index(self, fields, name=None, table=None, **kwargs):
        '''
        Build a new index on a cube.

        Examples:
            + index('field_name')

        :param fields: A single field or a list of (key, direction) pairs
        :param name: (optional) Custom name to use for this index
        :param collection: cube name
        :param owner: username of cube owner
        '''
        table = self.get_table(table)
        name = self._index_default_name(fields, name)
        fields = parse.parse_fields(fields)
        fields = self.columns(table, fields, reflect=True)
        session = self.session_new()
        index = Index(name, *fields)
        logger.info('Writing new index %s: %s' % (name, fields))
        result = index.create(self.engine)
        session.commit()
        return result