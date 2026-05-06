def geoip_data(self):
        """Attempt to retrieve MaxMind GeoIP data based on visitor's IP."""
        if not HAS_GEOIP or not TRACK_USING_GEOIP:
            return

        if not hasattr(self, '_geoip_data'):
            self._geoip_data = None
            try:
                gip = GeoIP(cache=GEOIP_CACHE_TYPE)
                self._geoip_data = gip.city(self.ip_address)
            except GeoIPException:
                msg = 'Error getting GeoIP data for IP "{0}"'.format(
                    self.ip_address)
                log.exception(msg)

        return self._geoip_data