def protocols(instance):
    """Ensure the 'protocols' property of network-traffic objects contains only
    values from the IANA Service Name and Transport Protocol Port Number
    Registry.
    """
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] == 'network-traffic' and
                'protocols' in obj):
            for prot in obj['protocols']:
                if enums.protocols():
                    if prot not in enums.protocols():
                        yield JSONError("The 'protocols' property of object "
                                        "'%s' contains a value ('%s') not in "
                                        "IANA Service Name and Transport "
                                        "Protocol Port Number Registry."
                                        % (key, prot), instance['id'],
                                        'protocols')
                else:
                    info("Can't reach IANA website; using regex for protocols.")
                    if not PROTOCOL_RE.match(prot):
                        yield JSONError("The 'protocols' property of object "
                                        "'%s' contains a value ('%s') not in "
                                        "IANA Service Name and Transport "
                                        "Protocol Port Number Registry."
                                        % (key, prot), instance['id'],
                                        'protocols')