def parse_config_list(config_list):
  """
  Parse a list of configuration properties separated by '='
  """
  if config_list is None:
    return {}
  else:
    mapping = {}
    for pair in config_list:
      if (constants.CONFIG_SEPARATOR not in pair) or (pair.count(constants.CONFIG_SEPARATOR) != 1):
        raise ValueError("configs must be passed as two strings separted by a %s", constants.CONFIG_SEPARATOR)
      (config, value) = pair.split(constants.CONFIG_SEPARATOR)
      mapping[config] = value
    return mapping