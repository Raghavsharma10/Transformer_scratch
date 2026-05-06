def count_async(self, limit=None, **q_options):
    """Count the number of query results, up to a limit.

    This is the asynchronous version of Query.count().
    """
    qry = self._fix_namespace()
    return qry._count_async(limit=limit, **q_options)