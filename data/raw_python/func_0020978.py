def defer_entity_syncing(wrapped, instance, args, kwargs):
    """
    A decorator that can be used to defer the syncing of entities until after the method has been run
    This is being introduced to help avoid deadlocks in the meantime as we attempt to better understand
    why they are happening
    """

    # Defer entity syncing while we run our method
    sync_entities.defer = True

    # Run the method
    try:
        return wrapped(*args, **kwargs)

    # After we run the method disable the deferred syncing
    # and sync all the entities that have been buffered to be synced
    finally:
        # Enable entity syncing again
        sync_entities.defer = False

        # Get the models that need to be synced
        model_objs = list(sync_entities.buffer.values())

        # If none is in the model objects we need to sync all
        if None in sync_entities.buffer:
            model_objs = list()

        # Sync the entities that were deferred if any
        if len(sync_entities.buffer):
            sync_entities(*model_objs)

        # Clear the buffer
        sync_entities.buffer = {}