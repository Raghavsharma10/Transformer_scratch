def scan(self, model_or_index, filter=None, projection="all", consistent=False, parallel=None):
        """Create a reusable :class:`~bloop.search.ScanIterator`.

        :param model_or_index: A model or index to scan.  For example, ``User`` or ``User.by_email``.
        :param filter: Filter condition.  Only matching objects will be included in the results.
        :param projection:
            "all", "count", a list of column names, or a list of :class:`~bloop.models.Column`.  When projection is
            "count", you must exhaust the iterator to retrieve the count.
        :param bool consistent: Use `strongly consistent reads`__ if True.  Default is False.
        :param tuple parallel: Perform a `parallel scan`__.  A tuple of (Segment, TotalSegments)
            for this portion the scan. Default is None.
        :return: A reusable scan iterator with helper methods.
        :rtype: :class:`~bloop.search.ScanIterator`

        __ http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.ReadConsistency.html
        __ http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/QueryAndScan.html#QueryAndScanParallelScan
        """
        if isinstance(model_or_index, Index):
            model, index = model_or_index.model, model_or_index
        else:
            model, index = model_or_index, None
        validate_not_abstract(model)
        s = Search(
            mode="scan", engine=self, model=model, index=index, filter=filter,
            projection=projection, consistent=consistent, parallel=parallel)
        return iter(s.prepare())