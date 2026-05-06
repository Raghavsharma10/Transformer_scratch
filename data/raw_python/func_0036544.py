def render(template, context, partials={}, state=None):
    """ Renders a given mustache template, with sane defaults. """
    # Create a new state by default
    state = state or State()

    # Add context to the state dict
    if isinstance(context, Context):
        state.context = context
    else:
        state.context = Context(context)

    # Add any partials to the state dict
    if partials:
        state.partials.push(partials)

    # Render the rendered template
    return __render(make_unicode(template), state)