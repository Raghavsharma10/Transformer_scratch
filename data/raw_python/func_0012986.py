def has_next_async(self):
    """Return a Future whose result will say whether a next item is available.

    See the module docstring for the usage pattern.
    """
    if self._fut is None:
      self._fut = self._iter.getq()
    flag = True
    try:
      yield self._fut
    except EOFError:
      flag = False
    raise tasklets.Return(flag)