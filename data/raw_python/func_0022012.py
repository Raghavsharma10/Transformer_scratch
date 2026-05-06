def _split_zone(server_name, zone_name):
    """
    Split the zone portion off from a DNS label.

    :param str server_name: The full DNS label.
    :param str zone_name: The zone name suffix.
    """
    server_name = server_name.rstrip(u'.')
    zone_name = zone_name.rstrip(u'.')
    if not (server_name == zone_name or
            server_name.endswith(u'.' + zone_name)):
        raise NotInZone(server_name=server_name, zone_name=zone_name)
    return server_name[:-len(zone_name)].rstrip(u'.')