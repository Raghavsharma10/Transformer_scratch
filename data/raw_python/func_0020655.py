def parse_config_file(config_file_path):
  """
  Parse a configuration file. Currently only supports .json, .py and properties separated by '='
  :param config_file_path:
  :return: a dict of the configuration properties
  """
  extension = os.path.splitext(config_file_path)[1]
  if extension == '.pyc':
    raise ValueError("Skipping .pyc file as config")
  if extension == '.json':
    with open(config_file_path) as config_file:
      try:
        mapping = json.load(config_file)
      except ValueError as e:
        logger.error("Did not load json configs", e)
        raise SyntaxError('Unable to parse config file:%s due to malformed JSON. Aborting' %(config_file_path))
  elif extension == '.py':
    mapping = {}
    file_dict = load_module(config_file_path)
    for attr_name in dir(file_dict):
      if not (attr_name.startswith('_') or attr_name.startswith('__')):
        attr = getattr(file_dict, attr_name)
        if type(attr) is dict:
          mapping.update(attr)
  else:
    with open(config_file_path) as config_file:
      lines = [line.rstrip() for line in config_file if line.rstrip() != "" and not line.startswith("#")]
      mapping = parse_config_list(lines)

  return mapping