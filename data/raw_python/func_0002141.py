def remote_access(self, service=None, use_xarray=None):
        """Access the remote dataset.

        Open the remote dataset and get a netCDF4-compatible `Dataset` object providing
        index-based subsetting capabilities.

        Parameters
        ----------
        service : str, optional
            The name of the service to use for access to the dataset, either
            'CdmRemote' or 'OPENDAP'. Defaults to 'CdmRemote'.

        Returns
        -------
        Dataset
            Object for netCDF4-like access to the dataset

        """
        if service is None:
            service = 'CdmRemote' if 'CdmRemote' in self.access_urls else 'OPENDAP'

        if service not in (CaseInsensitiveStr('CdmRemote'), CaseInsensitiveStr('OPENDAP')):
            raise ValueError(service + ' is not a valid service for remote_access')

        return self.access_with_service(service, use_xarray)