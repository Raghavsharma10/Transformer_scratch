def addToStore(store, identifier, name):
    """Adds a persisted factory with given identifier and object name to
    the given store.

    This is intended to have the identifier and name partially
    applied, so that a particular module with an exercise in it can
    just have an ``addToStore`` function that remembers it in the
    store.

    If a persisted factory with the same identifier already exists,
    the name will be updated.

    """
    persistedFactory = store.findOrCreate(_PersistedFactory, identifier=identifier)
    persistedFactory.name = name
    return persistedFactory