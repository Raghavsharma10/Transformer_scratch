def delete_entity_signal_handler(sender, instance, **kwargs):
    """
    Defines a signal handler for syncing an individual entity. Called when
    an entity is saved or deleted.
    """
    if instance.__class__ in entity_registry.entity_registry:
        Entity.all_objects.delete_for_obj(instance)