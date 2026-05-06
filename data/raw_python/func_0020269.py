def register(self, model, backend=None, read_backend=None,
                 include_related=True, **params):
        '''Register a :class:`Model` with this :class:`Router`. If the
model was already registered it does nothing.

:param model: a :class:`Model` class.
:param backend: a :class:`stdnet.BackendDataServer` or a
    :ref:`connection string <connection-string>`.
:param read_backend: Optional :class:`stdnet.BackendDataServer` for read
    operations. This is useful when the server has a master/slave
    configuration, where the master accept write and read operations
    and the ``slave`` read only operations.
:param include_related: ``True`` if related models to ``model`` needs to be
    registered. Default ``True``.
:param params: Additional parameters for the :func:`getdb` function.
:return: the number of models registered.
'''
        backend = backend or self._default_backend
        backend = getdb(backend=backend, **params)
        if read_backend:
            read_backend = getdb(read_backend)
        registered = 0
        if isinstance(model, Structure):
            self._structures[model] = StructureManager(model, backend,
                                                       read_backend, self)
            return model
        for model in models_from_model(model, include_related=include_related):
            if model in self._registered_models:
                continue
            registered += 1
            default_manager = backend.default_manager or Manager
            manager_class = getattr(model, 'manager_class', default_manager)
            manager = manager_class(model, backend, read_backend, self)
            self._registered_models[model] = manager
            if isinstance(model, ModelType):
                attr_name = model._meta.name
            else:
                attr_name = model.__name__.lower()
            if attr_name not in self._registered_names:
                self._registered_names[attr_name] = manager
            if self._install_global:
                model.objects = manager
        if registered:
            return backend