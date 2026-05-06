def decorator(func):
  r"""Makes the passed decorators to support optional args.
  """
  def wrapper(__decorated__=None, *Args, **KwArgs):
    if __decorated__ is None: # the decorator has some optional arguments.
      return lambda _func: func(_func, *Args, **KwArgs)

    else:
      return func(__decorated__, *Args, **KwArgs)

  return wrap(wrapper, func)