def _count_async(self, limit=None, **q_options):
    """Internal version of count_async()."""
    # TODO: Support offset by incorporating it to the limit.
    if 'offset' in q_options:
      raise NotImplementedError('.count() and .count_async() do not support '
                                'offsets at present.')
    if 'limit' in q_options:
      raise TypeError('Cannot specify limit as a non-keyword argument and as a '
                      'keyword argument simultaneously.')
    elif limit is None:
      limit = _MAX_LIMIT
    if self._needs_multi_query():
      # _MultiQuery does not support iterating over result batches,
      # so just fetch results and count them.
      # TODO: Use QueryIterator to avoid materializing the results list.
      q_options.setdefault('batch_size', limit)
      q_options.setdefault('keys_only', True)
      results = yield self.fetch_async(limit, **q_options)
      raise tasklets.Return(len(results))

    # Issue a special query requesting 0 results at a given offset.
    # The skipped_results count will tell us how many hits there were
    # before that offset without fetching the items.
    q_options['offset'] = limit
    q_options['limit'] = 0
    options = self._make_options(q_options)
    conn = tasklets.get_context()._conn
    dsquery = self._get_query(conn)
    rpc = dsquery.run_async(conn, options)
    total = 0
    while rpc is not None:
      batch = yield rpc
      options = QueryOptions(offset=options.offset - batch.skipped_results,
                             config=options)
      rpc = batch.next_batch_async(options)
      total += batch.skipped_results
    raise tasklets.Return(total)