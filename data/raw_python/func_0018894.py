def open_netcdf_writer(self, flatten=False, isolate=False, timeaxis=1):
        """Prepare a new |NetCDFInterface| object for writing data."""
        self._netcdf_writer = netcdftools.NetCDFInterface(
            flatten=bool(flatten),
            isolate=bool(isolate),
            timeaxis=int(timeaxis))