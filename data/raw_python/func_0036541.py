def get_match_info(template, match, state):
    """
    Given a template and a regex match within said template, return a
    dictionary of information about the match to be used to help parse the
    template.
    """
    info = match.groupdict()

    # Put special delimiter cases in terms of normal ones
    if info['change']:
        info.update({
            'tag_type' : '=',
            'tag_key' : info['delims'],
        })
    elif info['raw']:
        info.update({
            'tag_type' : '&',
            'tag_key' : info['raw_key'],
        })

    # Rename the important match variables for convenience
    tag_start = match.start()
    tag_end = match.end()
    tag_type = info['tag_type']
    tag_key = info['tag_key']
    lead_wsp = info['lead_wsp']
    end_wsp = info['end_wsp']

    begins_line = (tag_start == 0) or (template[tag_start-1] in state.eol_chars)
    ends_line = (tag_end == len(template) or
                 template[tag_end] in state.eol_chars)
    interpolating = (tag_type in ('', '&'))
    standalone = (not interpolating) and begins_line and ends_line

    if end_wsp:
        tag_end -= len(end_wsp)
    if standalone:
        template_length = len(template)
        # Standalone tags strip exactly one occurence of '\r', '\n', or '\r\n'
        # from the end of the line.
        if tag_end < len(template) and template[tag_end] == '\r':
            tag_end += 1
        if tag_end < len(template) and template[tag_end] == '\n':
            tag_end += 1
    elif lead_wsp:
        tag_start += len(lead_wsp)
        lead_wsp = ''

    info.update({
        'tag_start' : tag_start,
        'tag_end' : tag_end,
        'tag_type' : tag_type,
        'tag_key' : tag_key,
        'lead_wsp' : lead_wsp,
        'end_wsp' : end_wsp,
        'begins_line' : begins_line,
        'ends_line' : ends_line,
        'interpolating' : interpolating,
        'standalone' : standalone,
    })
    return info