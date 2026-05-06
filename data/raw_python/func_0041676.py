def reconfigArg(ArgConfig):
  r"""Reconfigures an argument based on its configuration.
  """
  _type = ArgConfig.get('type')

  if _type:
    if hasattr(_type, '__ec_config__'): # pass the ArgConfig to the CustomType:
      _type.__ec_config__(ArgConfig)

  if not 'type_str' in ArgConfig:
    ArgConfig['type_str'] = (_type.__name__ if isinstance(_type, type) else 'unspecified type') if _type else 'str'

  if not 'desc' in ArgConfig:
    ArgConfig['desc'] = ArgConfig['name']

  return ArgConfig