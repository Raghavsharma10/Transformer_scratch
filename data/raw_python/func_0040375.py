def start():
  r"""Starts ec.
  """
  processPendingModules()

  if not state.main_module_name in ModuleMembers: # don't start the core when main is not Ec-ed
    return

  MainModule = sys.modules[state.main_module_name]

  if not MainModule.__ec_member__.Members: # there was some error while loading script(s)
    return

  global BaseGroup
  BaseGroup = MainModule.__ec_member__

  Argv = sys.argv[1:]
  global mode
  mode = 'd' if Argv else 's' # dispatch / shell mode

  if mode == 's':
    import shell
    shell.init()

  else:
    import dispatch
    dispatch.init(Argv)

  processExitHooks()