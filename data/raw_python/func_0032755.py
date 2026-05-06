def request_openbus(self, service, endpoint, **kwargs):
        """Make a request to the given endpoint of the ``openbus`` server.

        This returns the plain JSON (dict) response which can then be parsed
        using one of the implemented types.

        Args:
            service (str): Service to fetch ('bus' or 'geo').
            endpoint (str): Endpoint to send the request to.
                This string corresponds to the key in the ``ENDPOINTS`` dict.
            **kwargs: Request arguments.

        Returns:
            Obtained response (dict) or None if the endpoint was not found.
        """
        if service == 'bus':
            endpoints = ENDPOINTS_BUS

        elif service == 'geo':
            endpoints = ENDPOINTS_GEO

        else:
            # Unknown service
            return None

        if endpoint not in endpoints:
            # Unknown endpoint
            return None

        url = URL_OPENBUS + endpoints[endpoint]

        # Append credentials to request
        kwargs['idClient'] = self._emt_id
        kwargs['passKey'] = self._emt_pass

        # SSL verification fails...
        # return requests.post(url, data=kwargs, verify=False).json()
        return requests.post(url, data=kwargs, verify=True).json()