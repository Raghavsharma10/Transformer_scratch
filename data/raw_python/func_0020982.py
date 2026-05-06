def sync_entities_watching(instance):
    """
    Syncs entities watching changes of a model instance.
    """
    for entity_model, entity_model_getter in entity_registry.entity_watching[instance.__class__]:
        model_objs = list(entity_model_getter(instance))
        if model_objs:
            sync_entities(*model_objs)