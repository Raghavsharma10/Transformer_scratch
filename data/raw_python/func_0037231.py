def via_nupnp():
    """ Use method 2 as described by the Philips guide """
    bridges_from_portal = parse_portal_json()
    logger.info('Portal returned %d Hue bridges(s).',
                 len(bridges_from_portal))
    # Confirm Portal gave an accessible bridge device by reading from the returned
    # location.  Should look like: http://192.168.0.1/description.xml
    found_bridges = {}
    for bridge in bridges_from_portal:
        serial, bridge_info = parse_description_xml(bridge[1])
        if serial:
            found_bridges[serial] = bridge_info

    logger.debug('%s', found_bridges)
    if found_bridges:
        return found_bridges
    else:
        raise DiscoveryError('Portal returned nothing')