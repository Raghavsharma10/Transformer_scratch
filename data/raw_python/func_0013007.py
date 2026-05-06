def _make_ctx_options(ctx_options, config_cls=ContextOptions):
  """Helper to construct a ContextOptions object from keyword arguments.

  Args:
    ctx_options: A dict of keyword arguments.
    config_cls: Optional Configuration class to use, default ContextOptions.

  Note that either 'options' or 'config' can be used to pass another
  Configuration object, but not both.  If another Configuration
  object is given it provides default values.

  Returns:
    A Configuration object, or None if ctx_options is empty.
  """
  if not ctx_options:
    return None
  for key in list(ctx_options):
    translation = _OPTION_TRANSLATIONS.get(key)
    if translation:
      if translation in ctx_options:
        raise ValueError('Cannot specify %s and %s at the same time' %
                         (key, translation))
      ctx_options[translation] = ctx_options.pop(key)
  return config_cls(**ctx_options)