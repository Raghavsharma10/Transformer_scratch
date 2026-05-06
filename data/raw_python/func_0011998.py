def get_api_services_by_name(self):
        """Return a dict of services by name"""
        if not self.services_by_name:
            self.services_by_name = dict({s.get('name'): s for s in self.conf
                                          .get("api")
                                          .get("services")})
        return self.services_by_name