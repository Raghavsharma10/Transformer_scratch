def get(type=None, **ArgConfig):
  r"""Helps to interactively get user input.

  Args:
    desc (str): The description for input.
    type (type / CustomType): The type of the input (defaults to None).
  """
  ArgConfig.update(type=type)
  return gatherInput(**reconfigArg(ArgConfig))