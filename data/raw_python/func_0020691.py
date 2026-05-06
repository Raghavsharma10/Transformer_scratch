def matches_ip(self, ip):
        """Return True if the given IP is blacklisted, False otherwise."""

        # Check the cache if caching is enabled
        if self.cache is not None:
            matches_ip = self.cache.get(ip)
            if matches_ip is not None:
                return matches_ip

        # Query MongoDB to see if the IP is blacklisted
        matches_ip = IPNetwork.matches_ip(
            ip, read_preference=self.read_preference)

        # Cache the result if caching is enabled
        if self.cache is not None:
            self.cache[ip] = matches_ip

        return matches_ip