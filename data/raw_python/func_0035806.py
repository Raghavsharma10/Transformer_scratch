def validate_settings(settings):
  """
  `settings` is either a dictionary or an object containing Kronos settings
  (e.g., the contents of conf/settings.py). This function checks that all
  required settings are present and valid.
  """

  # Validate `storage`
  storage = _validate_and_get_value(settings, 'settings', 'storage', dict)
  for name, options in storage.iteritems():
    if 'backend' not in options:
      raise ImproperlyConfigured(
          '`storage[\'{}\'] must contain a `backend` key'.format(name))

    path = options['backend']
    module, cls = path.rsplit('.', 1)
    module = import_module(module)
    if not hasattr(module, cls):
      raise NotImplementedError('`{}` not implemented.'.format(cls))
    validate_storage_settings(getattr(module, cls), options)

  # Validate `streams_to_backends`
  namespace_to_streams_configuration = _validate_and_get_value(
      settings, 'settings', 'namespace_to_streams_configuration', dict)
  for namespace, prefix_confs in namespace_to_streams_configuration.iteritems():
    if '' not in prefix_confs:
      raise ImproperlyConfigured(
          'Must specify backends for the null prefix')

    for prefix, options in prefix_confs.iteritems():
      if prefix != '':
        # Validate stream prefix.
        validate_stream(prefix)

      backends = _validate_and_get_value(
          options,
          "namespace_to_streams_configuration['{}']['{}']".format(namespace,
                                                                  prefix),
          'backends', dict)
      for backend in backends.keys():
        if backend not in storage:
          raise ImproperlyConfigured(
              "`{}` backend for `namespace_to_streams_configuration['{}']"
              "['{}']` is not configured in `storage`"
              .format(backend, namespace, prefix))

      read_backend = _validate_and_get_value(
          options,
          "namespace_to_streams_configuration['{}']['{}']".format(namespace,
                                                                  prefix),
          'read_backend', str)
      if read_backend not in storage:
          raise ImproperlyConfigured(
              "`{}` backend for `namespace_to_streams_configuration['{}']"
              "['{}']` is not configured in `storage`"
              .format(read_backend, namespace, prefix))

  # Validate `stream`
  stream = getattr(settings, 'stream', dict)
  _validate_and_get_value(stream, 'stream', 'format', re._pattern_type)

  # Validate `node`
  node = getattr(settings, 'node', dict)
  _validate_and_get_value(node, 'node', 'greenlet_pool_size', int)
  _validate_and_get_value(node, 'node', 'id', str)