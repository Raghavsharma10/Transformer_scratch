def arg(name=None, **Config): # wraps the _arg decorator, in order to allow unnamed args
  r"""A decorator to configure an argument of a task.

  Config:
    * name (str): The name of the arg. When ommited the agument will be identified through the order of configuration.
    * desc (str): The description of the arg (optional).
    * type (type, CustomType, callable): The alias for the task (optional).

  Notes:
    * It always follows a @task or an @arg.
  """
  if name is not None: # allow name as a positional arg
    Config['name'] = name

  return lambda decorated: _arg(decorated, **Config)