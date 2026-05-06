def _get_existing(driver, zone_name, server_name, validation):
    """
    Get existing validation records.
    """
    if zone_name is None:
        zones = sorted(
            (z for z
             in driver.list_zones()
             if server_name.rstrip(u'.')
                .endswith(u'.' + z.domain.rstrip(u'.'))),
            key=lambda z: len(z.domain),
            reverse=True)
        if len(zones) == 0:
            raise NotInZone(server_name=server_name, zone_name=None)
    else:
        zones = [
            z for z
            in driver.list_zones()
            if z.domain == zone_name]
        if len(zones) == 0:
            raise ZoneNotFound(zone_name=zone_name)
    zone = zones[0]
    subdomain = _split_zone(server_name, zone.domain)
    existing = [
        record for record
        in zone.list_records()
        if record.name == subdomain and
        record.type == 'TXT' and
        record.data == validation]
    return zone, existing, subdomain