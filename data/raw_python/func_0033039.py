def ip_address(self):
        """
        The IP address of the first interface listed in the droplet's
        ``networks`` field (ordering IPv4 before IPv6), or `None` if there
        are no interfaces
        """
        networks = self.get("networks", {})
        v4nets = networks.get("v4", [])
        v6nets = networks.get("v6", [])
        try:
            return (v4nets + v6nets)[0].ip_address
        except IndexError:
            return None