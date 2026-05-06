def query(self, model_or_index, key, filter=None, projection="all", consistent=False, forward=True):
        """Create a reusable :class:`~bloop.search.QueryIterator`.

        :param model_or_index: A model or index to query.  For example, ``User`` or ``User.by_email``.
        :param key:
            Key condition.  This must include an equality against the hash key, and optionally one
            of a restricted set of conditions on the range key.
        :param filter: Filter condition.  Only matching objects will be included in the results.
        :param projection:
            "all", "count", a list of column names, or a list of :class:`~bloop.models.Column`.  When projection is
            "count", you must advance the iterator to retrieve the count.
        :param bool consistent: Use `strongly consistent reads`__ if True.  Default is False.
        :param bool forward:  Query in ascending or descending order.  Default is True (ascending).

        :return: A reusable query iterator with helper methods.
        :rtype: :class:`~bloop.search.QueryIterator`

        __ http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.ReadConsistency.html
        """
        if isinstance(model_or_index, Index):
            model, index = model_or_index.model, model_or_index
        else:
            model, index = model_or_index, None
        validate_not_abstract(model)
        q = Search(
            mode="query", engine=self, model=model, index=index, key=key, filter=filter,
            projection=projection, consistent=consistent, forward=forward)
        return iter(q.prepare())