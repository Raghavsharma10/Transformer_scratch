def _arg(__decorated__, **Config):
  r"""The worker for the arg decorator.
  """
  if isinstance(__decorated__, tuple):  # this decorator is followed by another arg decorator
    __decorated__[1].insert(0, Config)
    return __decorated__

  else:
    return __decorated__, [Config]