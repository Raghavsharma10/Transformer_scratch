def map_async(self, callback, pass_batch_into_callback=None,
                merge_future=None, **q_options):
    """Map a callback function or tasklet over the query results.

    This is the asynchronous version of Query.map().
    """
    qry = self._fix_namespace()
    return tasklets.get_context().map_query(
        qry,
        callback,
        pass_batch_into_callback=pass_batch_into_callback,
        options=self._make_options(q_options),
        merge_future=merge_future)