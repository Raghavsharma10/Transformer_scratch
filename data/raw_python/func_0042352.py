def load_fetched_objects_into_contexts(events, model_data, context_hints_per_source):
    """
    Given the fetched model data and the context hints for each source, go through each
    event and populate the contexts with the loaded information.
    """
    for event in events:
        context_hints = context_hints_per_source.get(event.source, {})
        for context_key, hints in context_hints.items():
            model = get_model(hints['app_name'], hints['model_name'])
            for d, value in dict_find(event.context, context_key):
                if isinstance(value, list):
                    for i, model_id in enumerate(d[context_key]):
                        d[context_key][i] = model_data[model].get(model_id)
                else:
                    d[context_key] = model_data[model].get(value)