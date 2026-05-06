def register_service(self, info):
        """Registers service information to the network with a default TTL
        of 60 seconds.  Zeroconf will then respond to requests for
        information for that service.  The name of the service may be
        changed if needed to make it unique on the network."""
        self.check_service(info)
        self.services[info.name.lower()] = info

        # zone transfer
        self.transfer_zone(info.type)
        self.announce_service(info.name)