def get_tag_context(name, state):
    """
    Given a tag name, return its associated value as defined in the current
    context stack.
    """
    new_contexts = 0
    ctm = None
    while True:
        try:
            ctx_key, name = name.split('.', 1)
            ctm = state.context.get(ctx_key)
        except ValueError:
            break
        if not ctm:
            break
        else:
            state.context.push(ctm)
            new_contexts += 1

    ctm = state.context.get(name)

    return new_contexts, ctm