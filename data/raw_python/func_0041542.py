def wrap(func, with_func):
  r"""Copies the function signature from the wrapped function to the wrapping function.
  """
  func.__name__ = with_func.__name__
  func.__doc__ = with_func.__doc__
  func.__dict__.update(with_func.__dict__)

  return func