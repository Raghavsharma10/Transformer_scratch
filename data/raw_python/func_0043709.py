def watch_zone(self, zone_id):
        """ Add a zone to the watchlist.
        Zones on the watchlist will push all
        state changes (and those of the source they are currently connected to)
        back to the client """
        r = yield from self._send_cmd(
                "WATCH %s ON" % (zone_id.device_str(), ))
        self._watched_zones.add(zone_id)
        return r