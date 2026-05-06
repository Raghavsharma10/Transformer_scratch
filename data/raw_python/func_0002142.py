def subset(self, service=None):
        """Subset the dataset.

        Open the remote dataset and get a client for talking to ``service``.

        Parameters
        ----------
        service : str, optional
            The name of the service for subsetting the dataset. Defaults to 'NetcdfSubset'
            or 'NetcdfServer', in that order, depending on the services listed in the
            catalog.

        Returns
        -------
        a client for communicating using ``service``

        """
        if service is None:
            for serviceName in self.ncssServiceNames:
                if serviceName in self.access_urls:
                    service = serviceName
                    break
            else:
                raise RuntimeError('Subset access is not available for this dataset.')
        elif service not in self.ncssServiceNames:
            raise ValueError(service + ' is not a valid service for subset. Options are: '
                             + ', '.join(self.ncssServiceNames))

        return self.access_with_service(service)