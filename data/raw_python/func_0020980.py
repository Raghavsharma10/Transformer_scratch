def _get_model_objs_to_sync(model_ids_to_sync, model_objs_map, sync_all):
    """
    Given the model IDs to sync, fetch all model objects to sync
    """
    model_objs_to_sync = {}
    for ctype, model_ids_to_sync_for_ctype in model_ids_to_sync.items():
        model_qset = entity_registry.entity_registry.get(ctype.model_class()).queryset

        if not sync_all:
            model_objs_to_sync[ctype] = model_qset.filter(id__in=model_ids_to_sync_for_ctype)
        else:
            model_objs_to_sync[ctype] = [
                model_objs_map[ctype, model_id] for model_id in model_ids_to_sync_for_ctype
            ]

    return model_objs_to_sync