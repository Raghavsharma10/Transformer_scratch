def load_contexts_and_renderers(events, mediums):
    """
    Given a list of events and mediums, load the context model data into the contexts of the events.
    """
    sources = {event.source for event in events}
    rendering_styles = {medium.rendering_style for medium in mediums if medium.rendering_style}

    # Fetch the default rendering style and add it to the set of rendering styles
    default_rendering_style = get_default_rendering_style()
    if default_rendering_style:
        rendering_styles.add(default_rendering_style)

    context_renderers = ContextRenderer.objects.filter(
        Q(source__in=sources, rendering_style__in=rendering_styles) |
        Q(source_group_id__in=[s.group_id for s in sources], rendering_style__in=rendering_styles)).select_related(
            'source', 'rendering_style').prefetch_related('source_group__source_set')

    context_hints_per_source = get_context_hints_per_source(context_renderers)
    model_querysets = get_querysets_for_context_hints(context_hints_per_source)
    model_ids_to_fetch = get_model_ids_to_fetch(events, context_hints_per_source)
    model_data = fetch_model_data(model_querysets, model_ids_to_fetch)
    load_fetched_objects_into_contexts(events, model_data, context_hints_per_source)
    load_renderers_into_events(events, mediums, context_renderers, default_rendering_style)

    return events