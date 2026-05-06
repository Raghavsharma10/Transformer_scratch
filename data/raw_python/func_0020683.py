def reset_all():
  """
  Clear relevant globals to start fresh
  :return:
  """
  global _username
  global _password
  global _active_config
  global _active_tests
  global _machine_names
  global _deployers
  reset_deployers()
  reset_collector()
  _username = None
  _password = None
  _active_config = None
  _active_tests = {}
  _machine_names = defaultdict()