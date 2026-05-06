def search_account_domains(self, domain=None, latitude=None, longitude=None, name=None):
        """
        Search account domains.

        Returns a list of up to 5 matching account domains
        
        Partial match on name / domain are supported
        """
        path = {}
        data = {}
        params = {}

        # OPTIONAL - name
        """campus name"""
        if name is not None:
            params["name"] = name

        # OPTIONAL - domain
        """no description"""
        if domain is not None:
            params["domain"] = domain

        # OPTIONAL - latitude
        """no description"""
        if latitude is not None:
            params["latitude"] = latitude

        # OPTIONAL - longitude
        """no description"""
        if longitude is not None:
            params["longitude"] = longitude

        self.logger.debug("GET /api/v1/accounts/search with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/search".format(**path), data=data, params=params, no_data=True)