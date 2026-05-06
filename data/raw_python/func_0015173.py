def format_local_dap(dap, full=False, **kwargs):
    '''Formaqts information about the given local DAP in a human readable form to list of lines'''
    lines = []

    # Determining label width
    label_width = dapi.DapFormatter.calculate_offset(BASIC_LABELS)

    # Metadata
    lines.append(dapi.DapFormatter.format_meta(dap.meta, labels=BASIC_LABELS,
                                               offset=label_width, **kwargs))

    # Assistants
    lines.append('')
    lines.append(dapi.DapFormatter.format_assistants(dap.assistants))

    # Snippets
    if full:
        lines.append('')
        lines.append(dapi.DapFormatter.format_snippets(dap.snippets))

    # Supported platforms
    if 'supported_platforms' in dap.meta:
        lines.append('')
        lines.append(dapi.DapFormatter.format_platforms(dap.meta['supported_platforms']))

    lines.append()
    return lines