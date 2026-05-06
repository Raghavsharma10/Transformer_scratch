def from_tuple(cls, queries):
        """Create a ``Domain`` given a set of complex query tuples.

        Args:
            queries (iter): An iterator of complex queries. Each iteration
                should contain either:

                * A data-set compatible with :func:`~domain.Domain.add_query`
                * A string to switch the join type

                Example::

                    [('subject', 'Test1'),
                     'OR',
                     ('subject', 'Test2')',
                     ('subject', 'Test3')',
                     ]
                    # The above is equivalent to:
                    #    subject:'Test1' OR subject:'Test2' OR subject:'Test3'

                    [('modified_at', datetime(2017, 01, 01)),
                     ('status', 'active'),
                     ]
                    # The above is equivalent to:
                    #    modified_at:[2017-01-01T00:00:00Z TO *]
                    #    AND status:"active"

        Returns:
            Domain: A domain representing the input queries.
        """
        domain = cls()
        join_with = cls.AND
        for query in queries:
            if query in [cls.OR, cls.AND]:
                join_with = query
            else:
                domain.add_query(query, join_with)
        return domain