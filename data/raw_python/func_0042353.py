def load_renderers_into_events(events, mediums, context_renderers, default_rendering_style):
    """
    Given the events and the context renderers, load the renderers into the event objects
    so that they may be able to call the 'render' method later on.
    """
    # Make a mapping of source groups and rendering styles to context renderers. Do
    # the same for sources and rendering styles to context renderers
    source_group_style_to_renderer = {
        (cr.source_group_id, cr.rendering_style_id): cr
        for cr in context_renderers if cr.source_group_id
    }
    source_style_to_renderer = {
        (cr.source_id, cr.rendering_style_id): cr
        for cr in context_renderers if cr.source_id
    }

    for e in events:
        for m in mediums:
            # Try the following when loading a context renderer for a medium in an event.
            # 1. Try to look up the renderer based on the source group and medium rendering style
            # 2. If step 1 doesn't work, look up based on the source and medium rendering style
            # 3. If step 2 doesn't work, look up based on the source group and default rendering style
            # 4. if step 3 doesn't work, look up based on the source and default rendering style
            # If none of those steps work, this event will not be able to be rendered for the mediun
            cr = source_group_style_to_renderer.get((e.source.group_id, m.rendering_style_id))
            if not cr:
                cr = source_style_to_renderer.get((e.source_id, m.rendering_style_id))
            if not cr and default_rendering_style:
                cr = source_group_style_to_renderer.get((e.source.group_id, default_rendering_style.id))
            if not cr and default_rendering_style:
                cr = source_style_to_renderer.get((e.source_id, default_rendering_style.id))

            if cr:
                e._context_renderers[m] = cr