def get_service_marketplace(self, available=True, unavailable=False,
            deprecated=False):
        """
        Returns a list of service names.  Can return all services, just
        those supported by PredixPy, or just those not yet supported by
        PredixPy.

        :param available: Return the services that are
            available in PredixPy.  (Defaults to True)

        :param unavailable: Return the services that are not yet
            supported by PredixPy.  (Defaults to False)

        :param deprecated: Return the services that are
            supported by PredixPy but no longer available. (True)
        """

        supported = set(self.supported.keys())
        all_services = set(self.space.get_services())

        results = set()
        if available:
            results.update(supported)
        if unavailable:
            results.update(all_services.difference(supported))
        if deprecated:
            results.update(supported.difference(all_services))

        return list(results)