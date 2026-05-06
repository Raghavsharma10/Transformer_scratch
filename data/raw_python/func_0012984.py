def cursor_before(self):
    """Return the cursor before the current item.

    You must pass a QueryOptions object with produce_cursors=True
    for this to work.

    If there is no cursor or no current item, raise BadArgumentError.
    Before next() has returned there is no cursor.  Once the loop is
    exhausted, this returns the cursor after the last item.
    """
    if self._exhausted:
      return self.cursor_after()
    if isinstance(self._cursor_before, BaseException):
      raise self._cursor_before
    return self._cursor_before