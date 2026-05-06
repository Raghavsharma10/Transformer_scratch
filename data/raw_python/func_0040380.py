def _execCommand(Argv, collect_missing):
  r"""Worker of execCommand.
  """
  if not Argv:
    raise HandledException('Please specify a command!')

  RouteParts = Argv[0].split('/')
  Args, KwArgs = getDigestableArgs(Argv[1:])

  ResolvedMember = getDescendant(BaseGroup, RouteParts[:])

  if isinstance(ResolvedMember, Group):
    raise HandledException('Please specify a task.', Member=ResolvedMember)

  if not isinstance(ResolvedMember, Task):
    raise HandledException('No such task.', Member=BaseGroup)

  return ResolvedMember.__collect_n_call__(*Args, **KwArgs) if collect_missing else ResolvedMember(*Args, **KwArgs)