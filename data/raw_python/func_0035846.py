def _generate_storage_broker_lookup():
    """Return dictionary of available storage brokers."""
    storage_broker_lookup = dict()
    for entrypoint in iter_entry_points("dtool.storage_brokers"):
        StorageBroker = entrypoint.load()
        storage_broker_lookup[StorageBroker.key] = StorageBroker
    return storage_broker_lookup