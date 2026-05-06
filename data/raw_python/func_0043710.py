def unwatch_zone(self, zone_id):
        """ Remove a zone from the watchlist. """
        self._watched_zones.remove(zone_id)
        return (yield from
                self._send_cmd("WATCH %s OFF" % (zone_id.device_str(), )))