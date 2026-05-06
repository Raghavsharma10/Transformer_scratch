def sync_entities(*model_objs):
    """
    Syncs entities

    Args:
        model_objs (List[Model]): The model objects to sync. If empty, all entities will be synced
    """

    # Check if we are deferring processing
    if sync_entities.defer:
        # If we dont have any model objects passed add a none to let us know that we need to sync all
        if not model_objs:
            sync_entities.buffer[None] = None
        else:
            # Add each model obj to the buffer
            for model_obj in model_objs:
                sync_entities.buffer[(model_obj.__class__, model_obj.pk)] = model_obj

        # Return false that we did not do anything
        return False

    # Create a syncer and sync
    EntitySyncer(*model_objs).sync()