def _get_async(self, **q_options):
    """Internal version of get_async()."""
    res = yield self.fetch_async(1, **q_options)
    if not res:
      raise tasklets.Return(None)
    raise tasklets.Return(res[0])