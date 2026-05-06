def call(command, collect_missing=False, silent=True):
  r"""Calls a task, as if it were called from the command line.

  Args:
    command (str): A route followed by params (as if it were entered in the shell).
    collect_missing (bool): Collects any missing argument for the command through the shell. Defaults to False.

  Returns:
    The return value of the called command.
  """
  return (_execCommand if silent else execCommand)(shlex.split(command), collect_missing)