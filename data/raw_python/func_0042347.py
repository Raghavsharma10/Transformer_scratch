def get_context_hints_per_source(context_renderers):
    """
    Given a list of context renderers, return a dictionary of context hints per source.
    """
    # Merge the context render hints for each source as there can be multiple context hints for
    # sources depending on the render target. Merging them together involves combining select
    # and prefetch related hints for each context renderer
    context_hints_per_source = defaultdict(lambda: defaultdict(lambda: {
        'app_name': None,
        'model_name': None,
        'select_related': set(),
        'prefetch_related': set(),
    }))
    for cr in context_renderers:
        for key, hints in cr.context_hints.items() if cr.context_hints else []:
            for source in cr.get_sources():
                context_hints_per_source[source][key]['app_name'] = hints['app_name']
                context_hints_per_source[source][key]['model_name'] = hints['model_name']
                context_hints_per_source[source][key]['select_related'].update(hints.get('select_related', []))
                context_hints_per_source[source][key]['prefetch_related'].update(hints.get('prefetch_related', []))

    return context_hints_per_source