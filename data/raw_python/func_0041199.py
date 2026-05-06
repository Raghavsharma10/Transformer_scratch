def register_action(action):
  """
  Adds an action to the parser cli.

  :param action(BaseAction): a subclass of the BaseAction class
  """
  sub = _subparsers.add_parser(action.meta('cmd'), help=action.meta('help'))
  sub.set_defaults(cmd=action.meta('cmd'))
  for (name, arg) in action.props().items():
    sub.add_argument(arg.name, arg.flag, **arg.options)
    _actions[action.meta('cmd')] = action