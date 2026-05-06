def get_active_config(config_option, default=None):
  """
  gets the config value associated with the config_option or returns an empty string if the config is not found
  :param config_option:
  :param default: if not None, will be used
  :return: value of config. If key is not in config, then default will be used if default is not set to None. 
  Otherwise, KeyError is thrown.
  """
  return _active_config.mapping[config_option] if default is None else _active_config.mapping.get(config_option, default)