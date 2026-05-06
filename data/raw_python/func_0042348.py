def get_querysets_for_context_hints(context_hints_per_source):
    """
    Given a list of context hint dictionaries, return a dictionary
    of querysets for efficient context loading. The return value
    is structured as follows:

    {
        model: queryset,
        ...
    }
    """
    model_select_relateds = defaultdict(set)
    model_prefetch_relateds = defaultdict(set)
    model_querysets = {}
    for context_hints in context_hints_per_source.values():
        for hints in context_hints.values():
            model = get_model(hints['app_name'], hints['model_name'])
            model_querysets[model] = model.objects
            model_select_relateds[model].update(hints.get('select_related', []))
            model_prefetch_relateds[model].update(hints.get('prefetch_related', []))

    # Attach select and prefetch related parameters to the querysets if needed
    for model, queryset in model_querysets.items():
        if model_select_relateds[model]:
            queryset = queryset.select_related(*model_select_relateds[model])
        if model_prefetch_relateds[model]:
            queryset = queryset.prefetch_related(*model_prefetch_relateds[model])
        model_querysets[model] = queryset

    return model_querysets