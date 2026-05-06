def _getFuncArgs(func):
  r"""Gives the details on the args of the given func.

  Args:
    func (function): The function to get details on.
  """
  code = func.func_code
  Defaults = func.func_defaults

  nargs = code.co_argcount
  ArgNames = code.co_varnames[:nargs]

  Args = OrderedDict()
  argCount = len(ArgNames)
  defCount = len(Defaults) if Defaults else 0
  diff = argCount - defCount

  for i in range(0, diff):
    Args[ArgNames[i]] = {}

  for i in range(diff, argCount):
    Args[ArgNames[i]] = {'default': Defaults[i - diff]}

  return Args