def task(__decorated__=None, **Config):
  r"""A decorator to make tasks out of functions.

  Config:
    * name (str): The name of the task. Defaults to __decorated__.__name__.
    * desc (str): The description of the task (optional).
    * alias (str): The alias for the task (optional).
  """
  if isinstance(__decorated__, tuple):  # the task has some args
    _Task = Task(__decorated__[0], __decorated__[1], Config=Config)

  else:
    _Task = Task(__decorated__, [], Config)

  state.ActiveModuleMemberQ.insert(0, _Task)

  return _Task.Underlying