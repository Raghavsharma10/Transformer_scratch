def _get_super_entities_by_ctype(model_objs_by_ctype, model_ids_to_sync, sync_all):
    """
    Given model objects organized by content type and a dictionary of all model IDs that need
    to be synced, organize all super entity relationships that need to be synced.

    Ensure that the model_ids_to_sync dict is updated with any new super entities
    that need to be part of the overall entity sync
    """
    super_entities_by_ctype = defaultdict(lambda: defaultdict(list))  # pragma: no cover
    for ctype, model_objs_for_ctype in model_objs_by_ctype.items():
        entity_config = entity_registry.entity_registry.get(ctype.model_class())
        super_entities = entity_config.get_super_entities(model_objs_for_ctype, sync_all)
        super_entities_by_ctype[ctype] = {
            ContentType.objects.get_for_model(model_class, for_concrete_model=False): relationships
            for model_class, relationships in super_entities.items()
        }

        # Continue adding to the set of entities that need to be synced
        for super_entity_ctype, relationships in super_entities_by_ctype[ctype].items():
            for sub_entity_id, super_entity_id in relationships:
                model_ids_to_sync[ctype].add(sub_entity_id)
                model_ids_to_sync[super_entity_ctype].add(super_entity_id)

    return super_entities_by_ctype