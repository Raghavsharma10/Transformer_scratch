def ipfix(instance):
    """Ensure the 'ipfix' property of network-traffic objects contains only
    values from the IANA IP Flow Information Export (IPFIX) Entities Registry.
    """
    ipf_pattern = re.compile(r'^[a-z][a-zA-Z0-9]+')
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] == 'network-traffic' and
                'ipfix' in obj):
            for ipf in obj['ipfix']:
                if enums.ipfix():
                    if ipf not in enums.ipfix():
                        yield JSONError("The 'ipfix' property of object "
                                        "'%s' contains a key ('%s') not in "
                                        "IANA IP Flow Information Export "
                                        "(IPFIX) Entities Registry."
                                        % (key, ipf), instance['id'],
                                        'ipfix')
                else:
                    info("Can't reach IANA website; using regex for ipfix.")
                    if not ipf_pattern.match(ipf):
                        yield JSONError("The 'ipfix' property of object "
                                        "'%s' contains a key ('%s') not in "
                                        "IANA IP Flow Information Export "
                                        "(IPFIX) Entities Registry."
                                        % (key, ipf), instance['id'],
                                        'ipfix')