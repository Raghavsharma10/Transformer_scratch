def matches(self, object):
    """
    <Purpose>
      Return True if 'object' matches this schema, False if it doesn't.
      If the caller wishes to signal an error on a failed match, check_match()
      should be called, which will raise a 'exceptions.FormatError' exception.
    """

    try:
      self.check_match(object)
    except securesystemslib.exceptions.FormatError:
      return False
    else:
      return True