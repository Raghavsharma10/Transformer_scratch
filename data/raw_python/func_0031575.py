def join(self, source, op='LEFT JOIN', on=''):
        """
        Join `source`.

        >>> sc = SQLConstructor('main', ['c1', 'c2'])
        >>> sc.join('sub', 'JOIN', 'main.id = sub.id')
        >>> (sql, params, keys) = sc.compile()
        >>> sql
        'SELECT c1, c2 FROM main JOIN sub ON main.id = sub.id'

        It is possible to pass another `SQLConstructor` as a source.

        >>> sc = SQLConstructor('main', ['c1', 'c2'])
        >>> sc.add_or_matches('{0} = {1}', 'c1', [111])
        >>> subsc = SQLConstructor('sub', ['d1', 'd2'])
        >>> subsc.add_or_matches('{0} = {1}', 'd1', ['abc'])
        >>> sc.join(subsc, 'JOIN', 'main.id = sub.id')
        >>> sc.add_column('d1')
        >>> (sql, params, keys) = sc.compile()
        >>> print(sql)                     # doctest: +NORMALIZE_WHITESPACE
        SELECT c1, c2, d1 FROM main
        JOIN ( SELECT d1, d2 FROM sub WHERE (d1 = ?) )
        ON main.id = sub.id
        WHERE (c1 = ?)

        `params` is set appropriately to include parameters for joined
        source:

        >>> params
        ['abc', 111]

        Note that `subsc.compile` is called when `sc.join(subsc, ...)`
        is called.  Therefore, calling `subsc.add_<predicate>` does not
        effect `sc`.

        :type source: str or SQLConstructor
        :arg  source: table
        :type     op: str
        :arg      op: operation (e.g., 'JOIN')
        :type     on: str
        :arg      on: on clause.  `source` ("right" source) can be
                      referred using `{r}` formatting field.

        """
        if isinstance(source, SQLConstructor):
            (sql, params, _) = source.compile()
            self.join_params.extend(params)
            jsrc = '( {0} )'.format(sql)
            if source.table_alias:
                jsrc += ' AS ' + source.table_alias
                on = on.format(r=source.table_alias)
        else:
            jsrc = source
            on = on.format(r=source)
        constraint = 'ON {0}'.format(on) if on else ''
        self.join_source = ' '.join([self.join_source, op, jsrc, constraint])