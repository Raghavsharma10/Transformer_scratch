def run_to_queue(self, queue, conn, options=None, dsquery=None):
    """Run this query, putting entities into the given queue."""
    try:
      multiquery = self._maybe_multi_query()
      if multiquery is not None:
        yield multiquery.run_to_queue(queue, conn, options=options)
        return

      if dsquery is None:
        dsquery = self._get_query(conn)
      rpc = dsquery.run_async(conn, options)
      while rpc is not None:
        batch = yield rpc
        if (batch.skipped_results and
            datastore_query.FetchOptions.offset(options)):
          offset = options.offset - batch.skipped_results
          options = datastore_query.FetchOptions(offset=offset, config=options)
        rpc = batch.next_batch_async(options)
        for i, result in enumerate(batch.results):
          queue.putq((batch, i, result))
      queue.complete()

    except GeneratorExit:
      raise
    except Exception:
      if not queue.done():
        _, e, tb = sys.exc_info()
        queue.set_exception(e, tb)
      raise