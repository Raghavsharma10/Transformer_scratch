def get_model_ids_to_fetch(events, context_hints_per_source):
    """
    Obtains the ids of all models that need to be fetched. Returns a dictionary of models that
    point to sets of ids that need to be fetched. Return output is as follows:

    {
        model: [id1, id2, ...],
        ...
    }
    """
    number_types = (complex, float) + six.integer_types
    model_ids_to_fetch = defaultdict(set)

    for event in events:
        context_hints = context_hints_per_source.get(event.source, {})
        for context_key, hints in context_hints.items():
            for d, value in dict_find(event.context, context_key):
                values = value if isinstance(value, list) else [value]
                model_ids_to_fetch[get_model(hints['app_name'], hints['model_name'])].update(
                    v for v in values if isinstance(v, number_types)
                )

    return model_ids_to_fetch