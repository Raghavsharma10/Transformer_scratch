def next(self):
    """Iterator protocol: get next item or raise StopIteration."""
    if self._fut is None:
      self._fut = self._iter.getq()
    try:
      try:
        # The future result is set by this class's _extended_callback
        # method.
        # pylint: disable=unpacking-non-sequence
        (ent,
         self._cursor_before,
         self._cursor_after,
         self._more_results) = self._fut.get_result()
        return ent
      except EOFError:
        self._exhausted = True
        raise StopIteration
    finally:
      self._fut = None