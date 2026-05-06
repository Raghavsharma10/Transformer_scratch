def query(self, *queries, **kw):
        """
        Return a new S instance with query args combined with existing
        set in a must boolean query.

        :arg queries: instances of Q
        :arg kw: queries in the form of ``field__action=value``

        There are three special flags you can use:

        * ``must=True``: Specifies that the queries and kw queries
          **must match** in order for a document to be in the result.

          If you don't specify a special flag, this is the default.

        * ``should=True``: Specifies that the queries and kw queries
          **should match** in order for a document to be in the result.

        * ``must_not=True``: Specifies the queries and kw queries
          **must not match** in order for a document to be in the result.

        These flags work by putting those queries in the appropriate
        clause of an Elasticsearch boolean query.

        Examples:

        >>> s = S().query(foo='bar')
        >>> s = S().query(Q(foo='bar'))
        >>> s = S().query(foo='bar', bat__match='baz')
        >>> s = S().query(foo='bar', should=True)
        >>> s = S().query(foo='bar', should=True).query(baz='bat', must=True)

        Notes:

        1. Don't specify multiple special flags, but if you did, `should`
           takes precedence.
        2. If you don't specify any, it defaults to `must`.
        3. You can specify special flags in the
           :py:class:`elasticutils.Q`, too. If you're building your
           query incrementally, using :py:class:`elasticutils.Q` helps
           a lot.

        See the documentation on :py:class:`elasticutils.Q` for more
        details on composing queries with Q.

        See the documentation on :py:class:`elasticutils.S` for more
        details on adding support for more query types.

        """
        q = Q()
        for query in queries:
            q += query

        if 'or_' in kw:
            # Backwards compatibile with pre-0.7 version.
            or_query = kw.pop('or_')

            # or_query here is a dict of key/val pairs. or_ indicates
            # they're in a should clause, so we generate the
            # equivalent Q and then add it in.
            or_query['should'] = True
            q += Q(**or_query)

        q += Q(**kw)

        return self._clone(next_step=('query', q))