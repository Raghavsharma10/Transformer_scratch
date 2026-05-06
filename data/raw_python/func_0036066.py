def load_backends(self):
    """
    Loads all the backends setup in settings.py.
    """
    for name, backend_settings in settings.storage.iteritems():
      backend_path = backend_settings['backend']
      backend_module, backend_cls = backend_path.rsplit('.', 1)
      backend_module = import_module(backend_module)
      # Create an instance of the configured backend.
      backend_constructor = getattr(backend_module, backend_cls)
      self.backends[name] = backend_constructor(name,
                                                self.namespaces,
                                                **backend_settings)