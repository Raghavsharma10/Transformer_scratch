def run(*args, **kwargs):
  """
  Runs the parser and it executes the action handler with the provided arguments from the CLI.

  Also catches the BaseError interrupting the execution and showing the error message to the user.

  Default arguments comes from the cli args (sys.argv array) but we can force those arguments  when writing tests:

  .. code-block:: python

    parser.run(['build', '--path', '/custom-app-path'].split())
  
  .. code-block:: python

    parser.run('build --path /custom-app-path')
  """
  cmd = _parser.parse_args(*args, **kwargs)
  if hasattr(cmd, 'cmd') is False:
    return _parser.print_help()
  Action = _actions.get(cmd.cmd)
  action = Action()
  try:
    action(**{k:getattr(cmd, k) for k in action.props().keys()})
  except errors.BaseError as e:
    e.print_error()