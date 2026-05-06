def exit_hook(callable, once=True):
  r"""A decorator that makes the decorated function to run while ec exits.

  Args:
    callable (callable): The target callable.
    once (bool): Avoids adding a func to the hooks, if it has been added already. Defaults to True.

  Note:
    Hooks are processedd in a LIFO order.
  """
  if once and callable in ExitHooks:
    return

  ExitHooks.append(callable)