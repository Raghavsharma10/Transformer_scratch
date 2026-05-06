def open_netcdf_reader(self, flatten=False, isolate=False, timeaxis=1):
        """Prepare a new |NetCDFInterface| object for reading data."""
        self._netcdf_reader = netcdftools.NetCDFInterface(
            flatten=bool(flatten),
            isolate=bool(isolate),
            timeaxis=int(timeaxis))