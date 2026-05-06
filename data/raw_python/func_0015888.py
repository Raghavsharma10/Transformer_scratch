def query(self, *args):
        """
        Query a fulltext index by key and query or just a plain Lucene query,

        i1 = gdb.nodes.indexes.get('people',type='fulltext', provider='lucene')
        i1.query('name','do*')
        i1.query('name:do*')

        In this example, the last two line are equivalent.
        """
        if not args or len(args) > 2:
            raise TypeError('query() takes 2 or 3 arguments (a query or a key '
                            'and a query) (%d given)' % (len(args) + 1))
        elif len(args) == 1:
            query, = args
            return self.get('text').query(text_type(query))
        else:
            key, query = args
            index_key = self.get(key)
            if isinstance(query, string_types):
                return index_key.query(query)
            else:
                if query.fielded:
                    raise ValueError('Queries with an included key should '
                                     'not include a field.')
                return index_key.query(text_type(query))