def format_dap_from_dapi(name, version='', full=False):
    '''Formats information about given DAP from DAPI in a human readable form to list of lines'''
    lines = []
    m, d = _get_metadap_dap(name, version)

    if d:
        # Determining label width
        labels = BASIC_LABELS + ['average_rank'] # average_rank comes from m, not d
        if full:
            labels.extend(EXTRA_LABELS)
        label_width = dapi.DapFormatter.calculate_offset(labels)

        # Metadata
        lines += dapi.DapFormatter.format_meta_lines(d, labels=labels, offset=label_width)
        lines.append(dapi.DapFormatter.format_dapi_score(m, offset=label_width))

        if 'assistants' in d:
            # Assistants
            assistants = sorted([a for a in d['assistants'] if a.startswith('assistants')])
            lines.append('')
            for line in dapi.DapFormatter.format_assistants_lines(assistants):
                lines.append(line)

            # Snippets
            if full:
                snippets = sorted([a for a in d['assistants'] if a.startswith('snippets')])
                lines.append('')
                lines += dapi.DapFormatter.format_snippets(snippets)

        # Supported platforms
        if d.get('supported_platforms', ''):
            lines.append('')
            lines += dapi.DapFormatter.format_platforms(d['supported_platforms'])

        lines.append('')
    return lines