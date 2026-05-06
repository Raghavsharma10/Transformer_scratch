def __render_tag(info, state):
    """ Render an individual tag by making the appropriate replacement within
    the current context (if any). """
    new_contexts, context_match = get_tag_context(info['tag_key'], state)
    replacement = ''

    if context_match or context_match == 0:
        replacement = context_match
    elif info['tag_key'] == '.':
        replacement = state.context()
    else:
        replacement = ''

    # Call all callables / methods / lambdas / functions
    if replacement and callable(replacement):
        replacement = make_unicode(replacement())

        state.push_tags(state.default_tags)
        replacement = __render(template=replacement, state=state)
        state.pop_tags()

    for i in xrange(new_contexts): state.context.pop()

    if state.escape():
        return html_escape(replacement)
    return replacement