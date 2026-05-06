def _fetch_page_async(self, page_size, **q_options):
    """Internal version of fetch_page_async()."""
    q_options.setdefault('batch_size', page_size)
    q_options.setdefault('produce_cursors', True)
    it = self.iter(limit=page_size + 1, **q_options)
    results = []
    while (yield it.has_next_async()):
      results.append(it.next())
      if len(results) >= page_size:
        break
    try:
      cursor = it.cursor_after()
    except datastore_errors.BadArgumentError:
      cursor = None
    raise tasklets.Return(results, cursor, it.probably_has_next())