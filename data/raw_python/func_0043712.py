def enumerate_zones(self):
        """ Return a list of (zone_id, zone_name) tuples """
        zones = []
        for controller in range(1, 8):
            for zone in range(1, 17):
                zone_id = ZoneID(zone, controller)
                try:
                    name = yield from self.get_zone_variable(zone_id, 'name')
                    if name:
                        zones.append((zone_id, name))
                except CommandException:
                    break
        return zones