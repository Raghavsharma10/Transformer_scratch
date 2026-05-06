def __render(template, state, index=0):
    """
    Given a /template/ string, a parser /state/, and a starting
    offset (/index/), return the rendered version of the template.
    """
    # Find a Match
    match = state.tag_re.search(template, index)
    if not match:
        return template[index:]

    info = get_match_info(template, match, state)
    _pre = template[index : info['tag_start']] # template before the tag
    _tag = template[info['tag_start'] : info['tag_end']] # tag
    _continue = info['tag_end'] # the index at which to continue

    # Comment
    if info['tag_type'] == '!':
        # Comments are removed from output
        repl = ""

    # Delimiter change
    elif info['tag_type'] == '=':
        # Delimiters are changed; the tag is rendered as ""
        delimiters = re.split(r'\s*', info['tag_key'])
        new_tags = state.tags(_copy=True)
        new_tags['otag'], new_tags['ctag'] = map(re.escape, delimiters)
        state.push_tags(new_tags)
        repl = ""

    # Plain tag
    elif info['tag_type'] == '':
        repl = __render_tag(info, state)

    # Raw tag (should not be escaped)
    elif info['tag_type'] == '&':
        state.escape.push(False)
        repl = __render_tag(info, state)
        state.escape.pop()

    # Partial
    elif info['tag_type'] == '>':
        partial_name = info['tag_key']
        partial_template = None
        new_dir = None
        lead_wsp = re.compile(r'^(.)', re.M)
        repl = ''
        try:
            # Cached
            partial_template = state.partials()[partial_name]
        except (KeyError, IndexError):
            try:
                # Load the partial template from a file (if it exists)
                new_dir, filename = split(partial_name)
                if new_dir:
                    state.partials_dir.push(new_dir)

                partial_template = load_template(filename, state.abs_partials_dir, state.extension,
                                                 state.encoding, state.encoding_error)
            except (IOError):
                pass

        if partial_template:
            # Preserve indentation
            if info['standalone']:
                partial_template = lead_wsp.sub(info['lead_wsp']+r'\1', partial_template)

            # Update state
            state.partials.push(state.partials()) # XXX wtf is this shit?
            state.push_tags(state.default_tags)

            # Render the partial
            repl = __render(partial_template, state)

            # Restore state
            state.partials.pop()
            state.pop_tags()
            if new_dir:
                state.partials_dir.pop()

    # Section

    # TODO(peter): add a stop= index to __render so that template_to_inner does
    # not need to be constructed with [:] indexing, which is extremely
    # expensive.
    elif info['tag_type'] in ('#', '^'):
        otag_info = info
        ctag_info = section_end_info(template, info['tag_key'], state, _continue)

        # Don't want to parse beyond the end of the inner section, but
        # must include information on prior contents so that whitespace
        # is preserved correctly and inner tags are not marked as standalone.
        inner_start = otag_info['tag_end']
        inner_end = ctag_info['tag_start']
        _continue = ctag_info['tag_end']

        template_with_inner = template[:inner_end]

        new_contexts, ctm = get_tag_context(otag_info['tag_key'], state)
        truthy = otag_info['tag_type'] == '#'

        #if ctm is not None:
        if ctm:
            # If there's a match and it's callable, feed it the inner template
            if callable(ctm):
                template_to_inner = template[:inner_start]
                inner = template[inner_start:inner_end]
                template_with_inner = template_to_inner + make_unicode(ctm(inner))

            # Make the context list an iterable from the ctm
            if not hasattr(ctm, '__iter__') or isinstance(ctm, dict):
                ctx_list = [ctm]
            else:
                ctx_list = ctm
        # If there's no match, there are no new contexts
        else:
            ctx_list = [False]

        # If there are new contexts and the section is truthy, or if
        # there are no new contexts and the section is falsy, render
        # the contents
        repl_stack = []
        for ctx in ctx_list:
            if (truthy and ctx) or (not truthy and not ctx):
                state.context.push(ctx)
                repl_stack.append(
                    __render(template_with_inner, state, inner_start))

            else:
                break

        repl = ''.join(repl_stack)
        for i in xrange(new_contexts): state.context.pop()
    else:
        raise Exception("found unpaired end of section tag!")

    return u''.join((
        _pre, make_unicode(repl), __render(template, state, _continue)))