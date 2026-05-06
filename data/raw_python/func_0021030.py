def save_entity_signal_handler(sender, instance, **kwargs):
    """
    Defines a signal handler for saving an entity. Syncs the entity to
    the entity mirror table.
    """
    if instance.__class__ in entity_registry.entity_registry:
        sync_entities(instance)

    if instance.__class__ in entity_registry.entity_watching:
        sync_entities_watching(instance)