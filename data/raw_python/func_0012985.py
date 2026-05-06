def cursor_after(self):
    """Return the cursor after the current item.

    You must pass a QueryOptions object with produce_cursors=True
    for this to work.

    If there is no cursor or no current item, raise BadArgumentError.
    Before next() has returned there is no cursor.    Once the loop is
    exhausted, this returns the cursor after the last item.
    """
    if isinstance(self._cursor_after, BaseException):
      raise self._cursor_after
    return self._cursor_after