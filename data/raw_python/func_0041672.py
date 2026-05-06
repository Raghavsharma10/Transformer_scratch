def exit(exit_code=0):
  r"""A function to support exiting from exit hooks.

  Could also be used to exit from the calling scripts in a thread safe manner.
  """
  core.processExitHooks()

  if state.isExitHooked and not hasattr(sys, 'exitfunc'): # The function is called from the exit hook
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(exit_code) #pylint: disable=W0212

  sys.exit(exit_code)