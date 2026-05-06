def build_where_clause(mappings, operator='AND'):
        """Constructs the where clause based on a dictionary of values

        >>> build_where_clause({'id': 456, 'name': 'myrecord'}, operator='OR')
        >>> 'WHERE id = 456 OR name = "myrecord" '

        """
        where_clause_mappings = {}
        where_clause_mappings.update(mappings)

        where_clause = 'WHERE ' + ' {} '.format(operator).join(
            '{k} = {v}'.format(k=k, v='"{}"'.format(v) if isinstance(v, basestring) else v)
            for k, v in where_clause_mappings.iteritems()
        )
        return where_clause