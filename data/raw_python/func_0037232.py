def via_scan():
    """ IP scan - now implemented """
    import socket
    import ipaddress
    import httpfind
    bridges_from_scan = []
    hosts = socket.gethostbyname_ex(socket.gethostname())[2]
    for host in hosts:
        bridges_from_scan += httpfind.survey(
            # TODO: how do we determine subnet configuration?
            ipaddress.ip_interface(host+'/24').network,
            path='description.xml',
            pattern='(P|p)hilips')
        logger.info('Scan on %s', host)
    logger.info('Scan returned %d Hue bridges(s).', len(bridges_from_scan))
    # Confirm Scan gave an accessible bridge device by reading from the returned
    # location.  Should look like: http://192.168.0.1/description.xml
    found_bridges = {}
    for bridge in bridges_from_scan:
        serial, bridge_info = parse_description_xml(bridge)
        if serial:
            found_bridges[serial] = bridge_info

    logger.debug('%s', found_bridges)
    if found_bridges:
        return found_bridges
    else:
        raise DiscoveryError('Scan returned nothing')