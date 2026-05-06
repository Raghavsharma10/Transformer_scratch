def reassemble_options(payload):
    '''
    Reassemble partial options to options, returns a list of dhcp_option
    
    DHCP options are basically `|tag|length|value|` structure. When an
    option is longer than 255 bytes, it can be splitted into multiple
    structures with the same tag. The splitted structures must be
    joined back to get the original option.
    
    `dhcp_option_partial` is used to present the splitted options,
    and `dhcp_option` is used for reassembled option.
    '''
    options = []
    option_indices = {}
    def process_option_list(partials):
        for p in partials:
            if p.tag == OPTION_END:
                break
            if p.tag == OPTION_PAD:
                continue
            if p.tag in option_indices:
                # Reassemble the data
                options[option_indices[p.tag]][1].append(p.data)
            else:
                options.append((p.tag, [p.data]))
                option_indices[p.tag] = len(options) - 1
    # First process options field
    process_option_list(payload.options)
    if OPTION_OVERLOAD in option_indices:
        # There is an overload option
        data = b''.join(options[option_indices[OPTION_OVERLOAD]][1])
        overload_option = dhcp_overload.create(data)
        if overload_option & OVERLOAD_FILE:
            process_option_list(dhcp_option_partial[0].create(payload.file))
        if overload_option & OVERLOAD_SNAME:
            process_option_list(dhcp_option_partial[0].create(payload.sname))
    def _create_dhcp_option(tag, data):
        opt = dhcp_option(tag = tag)
        opt._setextra(data)
        opt._autosubclass()
        return opt
    return [_create_dhcp_option(tag, b''.join(data)) for tag,data in options]