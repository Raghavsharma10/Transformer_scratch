def validate_storage_settings(storage_class, settings):
  """
  Given a `storage_class` and a dictionary of `settings` to initialize it,
  this method verifies that all the settings are valid.
  """
  if not isinstance(settings, dict):
    raise ImproperlyConfigured(
        '{}: storage class settings must be a dict'.format(storage_class))

  if not hasattr(storage_class, 'SETTINGS_VALIDATORS'):
    raise NotImplementedError(
      '{}: storage class must define `SETTINGS_VALIDATORS`'.format(
        storage_class))

  settings_validators = getattr(storage_class, 'SETTINGS_VALIDATORS')
  settings = settings.copy()
  settings.pop('backend', None)  # No need to validate the `backend` key.
  invalid_settings = set(settings.keys()) - set(settings_validators.keys())
  if invalid_settings:
    raise ImproperlyConfigured(
        '{}: invalid settings: {}'.format(storage_class, invalid_settings))

  for setting, value in settings.iteritems():
    if not settings_validators[setting](value):
      raise ImproperlyConfigured(
          '{}: invalid value for {}'.format(storage_class, setting))