def via_upnp():
    """ Use SSDP as described by the Philips guide """
    ssdp_list = ssdp_discover("ssdp:all", timeout=5)
    #import pickle
    #with open("ssdp.pickle", "wb") as f:
        #pickle.dump(ssdp_list,f)
    bridges_from_ssdp = [u for u in ssdp_list if 'IpBridge' in u.server]
    logger.info('SSDP returned %d items with %d Hue bridges(s).',
                 len(ssdp_list), len(bridges_from_ssdp))
    # Confirm SSDP gave an accessible bridge device by reading from the returned
    # location.  Should look like: http://192.168.0.1:80/description.xml
    found_bridges = {}
    for bridge in bridges_from_ssdp:
        serial, bridge_info = parse_description_xml(bridge.location)
        if serial:
            found_bridges[serial] = bridge_info

    logger.debug('%s', found_bridges)
    if found_bridges:
        return found_bridges
    else:
        raise DiscoveryError('SSDP returned nothing')