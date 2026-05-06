def access_with_service(self, service, use_xarray=None):
        """Access the dataset using a particular service.

        Return an Python object capable of communicating with the server using the particular
        service. For instance, for 'HTTPServer' this is a file-like object capable of
        HTTP communication; for OPENDAP this is a netCDF4 dataset.

        Parameters
        ----------
        service : str
            The name of the service for accessing the dataset

        Returns
        -------
            An instance appropriate for communicating using ``service``.

        """
        service = CaseInsensitiveStr(service)
        if service == 'CdmRemote':
            if use_xarray:
                from .cdmr.xarray_support import CDMRemoteStore
                try:
                    import xarray as xr
                    provider = lambda url: xr.open_dataset(CDMRemoteStore(url))  # noqa: E731
                except ImportError:
                    raise ImportError('CdmRemote access needs xarray to be installed.')
            else:
                from .cdmr import Dataset as CDMRDataset
                provider = CDMRDataset
        elif service == 'OPENDAP':
            if use_xarray:
                try:
                    import xarray as xr
                    provider = xr.open_dataset
                except ImportError:
                    raise ImportError('xarray to be installed if `use_xarray` is True.')
            else:
                try:
                    from netCDF4 import Dataset as NC4Dataset
                    provider = NC4Dataset
                except ImportError:
                    raise ImportError('OPENDAP access needs netCDF4-python to be installed.')
        elif service in self.ncssServiceNames:
            from .ncss import NCSS
            provider = NCSS
        elif service == 'HTTPServer':
            provider = session_manager.urlopen
        else:
            raise ValueError(service + ' is not an access method supported by Siphon')

        try:
            return provider(self.access_urls[service])
        except KeyError:
            raise ValueError(service + ' is not available for this dataset')