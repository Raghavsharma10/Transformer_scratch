def from_path(path):
  """
  Selects and returns a build class based on project structure/config from a given path.

  :param path(str): required path argument to be used
  """
  for item in ref:
    build = ref[item]
    valid_ = build['is_valid']
    if valid_(path) is True:
      return build['builder'](path)
  raise errors.InvalidProjectStructure()