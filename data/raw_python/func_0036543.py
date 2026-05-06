def section_end_info(template, tag_key, state, index):
    """
    Given the tag key of an opening section tag, find the corresponding closing
    tag (if it exists) and return information about that match.
    """

    state.section.push(tag_key)
    match = None
    matchinfo = None
    search_index = index

    while state.section:
        match = state.tag_re.search(template, search_index)
        if not match:
            raise Exception("Open section %s never closed" % tag_key)

        matchinfo = get_match_info(template, match, state)

        # If we find a new section tag, add it to the stack and keep going
        if matchinfo['tag_type'] in ('#', '^'):
            state.section.push(matchinfo['tag_key'])
        # If we find a closing tag for the current section, 'close' it by
        # popping the stack
        elif matchinfo['tag_type'] == '/':
            if matchinfo['tag_key'] == state.section():
                state.section.pop()
            else:
                raise Exception(
                    'Unexpected section end: received %s, expected {{/%s}}' % (
                        repr(match.group(0)), tag_key))
        search_index = matchinfo['tag_end']

    return matchinfo