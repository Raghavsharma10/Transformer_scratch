def decorator(wrapped_decorator):
  """Converts a function into a decorator that optionally accepts keyword
  arguments in its declaration.

  Example usage:
    @utils.decorator
    def decorator(func, args, kwds, op1=None):
      ... apply op1 ...
      return func(*args, **kwds)

    # Form (1), vanilla
    @decorator
    foo(...)
      ...

    # Form (2), with options
    @decorator(op1=5)
    foo(...)
      ...

  Args:
    wrapped_decorator: A function that accepts positional args (func, args,
      kwds) and any additional supported keyword arguments.

  Returns:
    A decorator with an additional 'wrapped_decorator' property that is set to
  the original function.
  """
  def helper(_func=None, **options):
    def outer_wrapper(func):
      @wrapping(func)
      def inner_wrapper(*args, **kwds):
        return wrapped_decorator(func, args, kwds, **options)
      return inner_wrapper

    if _func is None:
      # Form (2), with options.
      return outer_wrapper

    # Form (1), vanilla.
    if options:
      # Don't allow @decorator(foo, op1=5).
      raise TypeError('positional arguments not supported')
    return outer_wrapper(_func)
  helper.wrapped_decorator = wrapped_decorator
  return helper