def call_fn(fn, arglist, env):
  """Call a function, respecting all the various types of functions that exist."""
  if isinstance(fn, framework.LazyFunction):
    # The following looks complicated, but this is necessary because you can't
    # construct closures over the loop variable directly.
    thunks = [(lambda thunk: lambda: framework.eval(thunk, env))(th) for th in arglist.values]
    return fn(*thunks)

  evaled_args = framework.eval(arglist, env)
  if isinstance(fn, framework.EnvironmentFunction):
    return fn(*evaled_args, env=env)

  return fn(*evaled_args)