def compact_error(err):
  """Return the the last 2 error messages from an error stack.

  These error messages turns out to be the most descriptive.
  """
  def err2(e):
    if isinstance(e, exceptions.EvaluationError) and e.inner:
      message, i = err2(e.inner)
      if i == 1:
        return ', '.join([e.args[0], str(e.inner)]), i + 1
      else:
        return message, i + 1
    else:
      return str(e), 1
  return err2(err)[0]