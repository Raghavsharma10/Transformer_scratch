def fetch_async(self, limit=None, **q_options):
    """Fetch a list of query results, up to a limit.

    This is the asynchronous version of Query.fetch().
    """
    if limit is None:
      default_options = self._make_options(q_options)
      if default_options is not None and default_options.limit is not None:
        limit = default_options.limit
      else:
        limit = _MAX_LIMIT
    q_options['limit'] = limit
    q_options.setdefault('batch_size', limit)
    if self._needs_multi_query():
      return self.map_async(None, **q_options)
    # Optimization using direct batches.
    options = self._make_options(q_options)
    qry = self._fix_namespace()
    return qry._run_to_list([], options=options)