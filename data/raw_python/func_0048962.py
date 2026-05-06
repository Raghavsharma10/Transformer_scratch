def _validate_storage(storage, service_name, add_error):
    """Lazily validate the storage constraints, ensuring that they are a dict.

    Use the given add_error callable to register validation error.
    """
    if storage is None:
        return
    if not isdict(storage):
        msg = 'service {} has invalid storage constraints {}'.format(
            service_name, storage)
        add_error(msg)