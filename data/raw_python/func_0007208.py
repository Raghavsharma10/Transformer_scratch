def remove_zone_record(self, id, domain, subdomain=None):
        """
        Remove the zone record with the given ID that belongs to the given
        domain and sub domain. If no sub domain is given the wildcard sub-domain
        is assumed.
        """

        if subdomain is None:
            subdomain = "@"

        _validate_int("id", id)

        self._call("removeZoneRecord", domain, subdomain, id)