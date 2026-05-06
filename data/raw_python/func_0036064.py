def add_query(self, query, join_with=AND):
        """Join a new query to existing queries on the stack.

        Args:
            query (tuple or list or DomainCondition): The condition for the
                query. If a ``DomainCondition`` object is not provided, the
                input should conform to the interface defined in
                :func:`~.domain.DomainCondition.from_tuple`.
            join_with (str): The join string to apply, if other queries are
                already on the stack.
        """
        if not isinstance(query, DomainCondition):
            query = DomainCondition.from_tuple(query)
        if len(self.query):
            self.query.append(join_with)
        self.query.append(query)