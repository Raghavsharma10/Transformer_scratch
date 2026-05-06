def getTaskHelp(_Task):
  r"""Gets help on the given task member.
  """
  Ret = []

  for k in ['name', 'desc']:
    v = _Task.Config.get(k)

    if v is not None:
      Ret.append('%s: %s' % (k, v))

  Args = _Task.Args

  if Args:
    Ret.append('\nArgs:')

    for argName, Arg in Args.items():
      Ret.append('  %s: %s' % (argName, Arg.get('desc', Arg['type_str'])))

    Ret.append('')

  return '\n'.join(Ret).rstrip()