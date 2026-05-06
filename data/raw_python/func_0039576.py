def registerExitCall():
  r"""Registers an exit call to start the core.

    The core would be started after the main module is loaded. Ec would be exited from the core.
  """
  if state.isExitHooked:
    return

  state.isExitHooked = True

  from atexit import register

  register(core.start)